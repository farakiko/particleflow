import csv
import os
import pickle as pkl
import time
from pathlib import Path

import numpy as np
import torch

# import torch.nn as nn
import tqdm
from pyg.logger import _logger

# from pyg.training import FocalLoss
from pyg.utils import (  # unpack_predictions,
    get_model_state_dict,
    save_checkpoint,
    unpack_target,
)
from torch.utils.tensorboard import SummaryWriter

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


def configure_model_trainable(model, trainable, is_training):
    if is_training:
        model.train()
        if trainable != "all":
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            for layer in trainable:
                layer = getattr(model, layer)
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True
    else:
        model.eval()


def train_and_valid(
    rank,
    outdir,
    model,
    optimizer,
    train_loader,
    valid_loader,
    trainable,
    is_train=True,
    lr_schedule=None,
    epoch=None,
    val_freq=None,
    dtype=torch.float32,
    tensorboard_writer=None,
):

    configure_model_trainable(model, trainable, is_train)

    """
    Performs training over a given epoch. Will run a validation step every val_freq.
    """

    train_or_valid = "train" if is_train else "valid"
    _logger.info(f"Initiating epoch #{epoch} {train_or_valid} run on device rank={rank}", color="red")

    # this one will keep accumulating `train_loss` and then return the average
    epoch_loss = {}

    if is_train:
        data_loader = train_loader
    else:
        data_loader = valid_loader

    print("LEN", len(data_loader))
    # only show progress bar on rank 0
    iterator = tqdm.tqdm(
        enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} {train_or_valid} loop on rank={rank}"
    )

    loss = {}
    loss_accum = 0.0
    val_freq_time_0 = time.time()
    val_freq_step = 0
    for itrain, batch in iterator:

        batch = batch.to(rank, non_blocking=True)
        ygen = unpack_target(batch.ygen)
        ycand = unpack_target(batch.ycand)

        # DeepMET inference
        msk_ycand = ycand["cls_id"] != 0
        cand_px = (ycand["pt"] * ycand["cos_phi"]) * msk_ycand
        cand_py = (ycand["pt"] * ycand["sin_phi"]) * msk_ycand
        p4_masked = ycand["momentum"] * msk_ycand.unsqueeze(-1)

        if is_train:
            wx, wy = model(p4_masked)
        else:
            with torch.no_grad():
                wx, wy = model(p4_masked)

        pred_met = (wx * (torch.sum(cand_px, axis=1) ** 2)) + (wy * (torch.sum(cand_py, axis=1) ** 2))

        # genMET
        msk_gen = ygen["cls_id"] != 0
        gen_px = (ygen["pt"] * ygen["cos_phi"]) * msk_gen
        gen_py = (ygen["pt"] * ygen["sin_phi"]) * msk_gen

        true_met = torch.sum(gen_px, axis=1) ** 2 + torch.sum(gen_py, axis=1) ** 2

        if is_train:
            loss["MET"] = torch.nn.functional.huber_loss(pred_met, true_met)
            for param in model.parameters():
                param.grad = None
            loss["MET"].backward()
            loss_accum += loss["MET"].detach().cpu().item()
            optimizer.step()
        else:
            with torch.no_grad():
                loss["MET"] = torch.nn.functional.huber_loss(pred_met, true_met)

        for loss_ in loss.keys():
            if loss_ not in epoch_loss:
                epoch_loss[loss_] = 0.0
            epoch_loss[loss_] += loss[loss_].detach()

        if is_train:
            step = (epoch - 1) * len(data_loader) + itrain
            if not (tensorboard_writer is None):
                tensorboard_writer.add_scalar("step/loss_train", loss_accum, step)
                if itrain % 10 == 0:
                    tensorboard_writer.flush()
                loss_accum = 0.0

        if val_freq is not None and is_train:

            if itrain != 0 and itrain % val_freq == 0:
                # time since last intermediate validation run
                val_freq_time = torch.tensor(time.time() - val_freq_time_0, device=rank)
                # compute intermediate training loss
                intermediate_losses_t = {key: epoch_loss[key] for key in epoch_loss}
                for loss_ in epoch_loss:
                    intermediate_losses_t[loss_] = intermediate_losses_t[loss_].cpu().item() / itrain

                # compute intermediate validation loss
                intermediate_losses_v = train_and_valid(
                    rank,
                    outdir,
                    model,
                    optimizer,
                    train_loader,
                    valid_loader,
                    trainable,
                    is_train=False,
                    lr_schedule=None,
                    epoch=epoch,
                    val_freq=None,
                    dtype=dtype,
                )
                print("intermediate_losses_v", intermediate_losses_v)
                intermediate_metrics = dict(
                    loss=intermediate_losses_t["MET"],
                    val_loss=intermediate_losses_v["MET"],
                    inside_epoch=epoch,
                    step=(epoch - 1) * len(data_loader) + itrain,
                    val_freq_time=val_freq_time.cpu().item(),
                )
                val_freq_log = os.path.join(outdir, "val_freq_log.csv")
                with open(val_freq_log, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=intermediate_metrics.keys())
                    if os.stat(val_freq_log).st_size == 0:  # only write header if file is empty
                        writer.writeheader()
                    writer.writerow(intermediate_metrics)
                val_freq_time_0 = time.time()  # reset intermediate validation spacing timer

                step = (epoch - 1) * len(data_loader) + itrain
                if not (tensorboard_writer is None):
                    tensorboard_writer.add_scalar("step/loss_intermediate_t", intermediate_losses_t["MET"], val_freq_step)
                    tensorboard_writer.add_scalar("step/loss_intermediate_v", intermediate_losses_v["MET"], val_freq_step)
                val_freq_step += 1

        if not is_train:
            if itrain > 10:
                break

    for loss_ in epoch_loss:
        epoch_loss[loss_] = epoch_loss[loss_].cpu().item() / len(data_loader)

    return epoch_loss


def train_mlpf(
    rank,
    model,
    optimizer,
    train_loader,
    valid_loader,
    num_epochs,
    patience,
    outdir,
    trainable="all",
    dtype=torch.float32,
    checkpoint_freq=None,
    val_freq=None,
):
    """
    Will run a full training by calling train().

    Args:
        rank: 'cpu' or int representing the gpu device id
        model: a pytorch model (may be wrapped by DistributedDataParallel)
        train_loader: a pytorch geometric Dataloader that loads the training data in the form ~ DataBatch(X, ygen, ycands)
        valid_loader: a pytorch geometric Dataloader that loads the validation data in the form ~ DataBatch(X, ygen, ycands)
        patience: number of stale epochs before stopping the training
        outdir: path to store the model weights and training plots
    """

    global val_freq_step
    val_freq_step = 0

    tensorboard_writer_train = SummaryWriter(f"{outdir}/runs/train")
    tensorboard_writer_valid = SummaryWriter(f"{outdir}/runs/valid")

    t0_initial = time.time()

    losses_of_interest = ["MET"]

    losses = {}
    losses["train"], losses["valid"] = {}, {}
    for loss in losses_of_interest:
        losses["train"][loss], losses["valid"][loss] = [], []

    stale_epochs, best_val_loss = torch.tensor(0, device=rank), float("inf")

    start_epoch = 1
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        losses_t = train_and_valid(
            rank,
            outdir,
            model,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=True,
            epoch=epoch,
            dtype=dtype,
            val_freq=val_freq,
            tensorboard_writer=tensorboard_writer_train,
        )

        losses_v = train_and_valid(
            rank,
            outdir,
            model,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=False,
            epoch=epoch,
            val_freq=val_freq,
            dtype=dtype,
        )

        extra_state = {"epoch": epoch}
        if losses_v["MET"] < best_val_loss:
            best_val_loss = losses_v["MET"]
            stale_epochs = 0
            torch.save(
                {"model_state_dict": get_model_state_dict(model), "optimizer_state_dict": optimizer.state_dict()},
                f"{outdir}/best_weights.pth",
            )
            save_checkpoint(f"{outdir}/best_weights.pth", model, optimizer, extra_state)
        else:
            stale_epochs += 1

        if checkpoint_freq and (epoch != 0) and (epoch % checkpoint_freq == 0):
            checkpoint_dir = Path(outdir) / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = "{}/checkpoint-{:02d}-{:.6f}.pth".format(checkpoint_dir, epoch, losses_v["MET"])
            save_checkpoint(checkpoint_path, model, optimizer, extra_state)

        if stale_epochs > patience:
            break

        for k, v in losses_t.items():
            tensorboard_writer_train.add_scalar("epoch/loss_" + k, v, epoch)

        for loss in losses_of_interest:
            losses["train"][loss].append(losses_t[loss])
            losses["valid"][loss].append(losses_v[loss])

        for k, v in losses_v.items():
            tensorboard_writer_valid.add_scalar("epoch/loss_" + k, v, epoch)

        t1 = time.time()

        epochs_remaining = num_epochs - epoch
        time_per_epoch = (t1 - t0_initial) / epoch
        eta = epochs_remaining * time_per_epoch / 60

        _logger.info(
            f"Rank {rank}: epoch={epoch} / {num_epochs} "
            + f"train_loss={losses_t['MET']:.4f} "
            + f"valid_loss={losses_v['MET']:.4f} "
            + f"stale={stale_epochs} "
            + f"time={round((t1-t0)/60, 2)}m "
            + f"eta={round(eta, 1)}m",
            color="bold",
        )

        with open(f"{outdir}/mlpf_losses.pkl", "wb") as f:
            pkl.dump(losses, f)

        if tensorboard_writer_train:
            tensorboard_writer_train.flush()
        if tensorboard_writer_valid:
            tensorboard_writer_valid.flush()

    _logger.info(f"Done with training. Total training time on device {rank} is {round((time.time() - t0_initial)/60,3)}min")


def override_config(config, args):
    """override config with values from argparse Namespace"""
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if arg_value is not None:
            config[arg] = arg_value

    if not (args.attention_type is None):
        config["model"]["attention"]["attention_type"] = args.attention_type

    if not (args.num_convs is None):
        for model in ["gnn_lsh", "gravnet", "attention", "attention", "mamba"]:
            config["model"][model]["num_convs"] = args.num_convs

    args.test_datasets = config["test_dataset"]

    return config
