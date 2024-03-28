# import csv
# import os
import pickle as pkl
import time
from pathlib import Path

import numpy as np
import torch

# import torch.nn as nn
import tqdm
from pyg.logger import _logger

# from pyg.training import FocalLoss
from pyg.utils import (
    get_model_state_dict,
    save_checkpoint,
    unpack_predictions,
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
    deepmet,
    which_deepmet,
    mlpf,
    optimizer,
    train_loader,
    valid_loader,
    trainable,
    is_train=True,
    lr_schedule=None,
    epoch=None,
    val_freq=None,
    val_freq_step=0,
    dtype=torch.float32,
    tensorboard_writer=None,
):
    """
    Performs training over a given epoch. Will run a validation step every val_freq.
    """

    train_or_valid = "train" if is_train else "valid"
    _logger.info(f"Initiating epoch #{epoch} {train_or_valid} run on device rank={rank}", color="red")

    configure_model_trainable(deepmet, trainable, is_train)

    # this one will keep accumulating `train_loss` and then return the average
    epoch_loss = {}

    if is_train:
        data_loader = train_loader
    else:
        data_loader = valid_loader

    # only show progress bar on rank 0
    iterator = tqdm.tqdm(
        enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} {train_or_valid} loop on rank={rank}"
    )

    loss = {}
    train_loss_accum = 0.0
    for itrain, batch in iterator:

        batch = batch.to(rank, non_blocking=True)
        ygen = unpack_target(batch.ygen)
        ycand = unpack_target(batch.ycand)

        # first check if MLPF inference must be done
        if mlpf == {}:  # use PF-cands
            ypred = ycand
        else:  # run the MLPF inference to get the MLPF cands
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                with torch.no_grad():
                    ymlpf = mlpf(batch.X, batch.mask)
            ymlpf = unpack_predictions(ymlpf)
            ypred = ymlpf

        msk_ypred = ypred["cls_id"] != 0
        pred_px = (ypred["pt"] * ypred["cos_phi"]) * msk_ypred
        pred_py = (ypred["pt"] * ypred["sin_phi"]) * msk_ypred
        p4_masked = ypred["momentum"] * msk_ypred.unsqueeze(-1)

        # runs the DeepMET inference
        if is_train:
            wx, wy = deepmet(p4_masked)
        else:
            with torch.no_grad():
                wx, wy = deepmet(p4_masked)

        if which_deepmet == "1":
            # pred_met = (torch.sum(wx * cand_px, axis=1) ** 2) + (torch.sum(wy * cand_py, axis=1) ** 2)
            pred_met_x = torch.sum(wx * pred_px, axis=1)
            pred_met_y = torch.sum(wy * pred_py, axis=1)
        else:
            # pred_met = (wx * (torch.sum(cand_px, axis=1) ** 2)) + (wy * (torch.sum(cand_py, axis=1) ** 2))
            pred_met_x = wx * torch.sum(pred_px, axis=1)
            pred_met_y = wy * torch.sum(pred_py, axis=1)

        # genMET to compute the loss
        msk_gen = ygen["cls_id"] != 0
        gen_px = (ygen["pt"] * ygen["cos_phi"]) * msk_gen
        gen_py = (ygen["pt"] * ygen["sin_phi"]) * msk_gen

        # true_met = torch.sum(gen_px, axis=1) ** 2 + torch.sum(gen_py, axis=1) ** 2
        true_met_x = torch.sum(gen_px, axis=1)
        true_met_y = torch.sum(gen_py, axis=1)

        if is_train:
            # loss["MET"] = torch.nn.functional.huber_loss(pred_met, true_met)
            loss["MET"] = torch.nn.functional.huber_loss(true_met_x, pred_met_x) + torch.nn.functional.huber_loss(
                true_met_y, pred_met_y
            )

            for param in deepmet.parameters():
                param.grad = None
            loss["MET"].backward()
            optimizer.step()
            train_loss_accum += loss["MET"].detach().cpu().item()
        else:
            with torch.no_grad():
                # loss["MET"] = torch.nn.functional.huber_loss(pred_met, true_met)
                loss["MET"] = torch.nn.functional.huber_loss(true_met_x, pred_met_x) + torch.nn.functional.huber_loss(
                    true_met_y, pred_met_y
                )

        for loss_ in loss.keys():
            if loss_ not in epoch_loss:
                epoch_loss[loss_] = 0.0
            epoch_loss[loss_] += loss[loss_].detach()

        # if val_freq is not None and is_train:

        #     if itrain != 0 and itrain % val_freq == 0:

        #         # compute intermediate training loss
        #         intermediate_losses_t = {"MET": train_loss_accum / val_freq}

        #         # compute intermediate validation loss
        #         intermediate_losses_v, _ = train_and_valid(
        #             rank,
        #             outdir,
        #             deepmet,
        #             which_deepmet,
        #             mlpf,
        #             optimizer,
        #             train_loader,
        #             valid_loader,
        #             trainable,
        #             is_train=False,
        #             lr_schedule=None,
        #             epoch=epoch,
        #             val_freq_step=None,
        #             dtype=dtype,
        #         )

        #         intermediate_metrics = dict(
        #             loss=intermediate_losses_t["MET"],
        #             val_loss=intermediate_losses_v["MET"],
        #             inside_epoch=epoch,
        #             step=val_freq_step,
        #         )
        #         val_freq_log = os.path.join(outdir, "val_freq_log.csv")
        #         with open(val_freq_log, "a", newline="") as f:
        #             writer = csv.DictWriter(f, fieldnames=intermediate_metrics.keys())
        #             if os.stat(val_freq_log).st_size == 0:  # only write header if file is empty
        #                 writer.writeheader()
        #             writer.writerow(intermediate_metrics)

        #         if not (tensorboard_writer is None):
        #             tensorboard_writer.add_scalar("step/loss_intermediate_t", intermediate_losses_t["MET"], val_freq_step)
        #             tensorboard_writer.add_scalar("step/loss_intermediate_v", intermediate_losses_v["MET"], val_freq_step)

        #         val_freq_step += 1

        #         # save checkpoint every validation step
        #         extra_state = {"epoch": epoch}
        #         checkpoint_dir = Path(outdir) / "checkpoints_val_freq"
        #         checkpoint_dir.mkdir(exist_ok=True)
        #         checkpoint_path = "{}/checkpoint-{:02d}-{:.6f}.pth".format(
        #             checkpoint_dir, val_freq_step, intermediate_losses_v["MET"]
        #         )
        #         save_checkpoint(checkpoint_path, deepmet, optimizer, extra_state)

        # use 300 to stop the validation frequency iteration
        if not is_train:
            if itrain > 300:
                break

    if not is_train:
        for loss_ in epoch_loss:
            epoch_loss[loss_] = epoch_loss[loss_].cpu().item() / 300

    else:
        for loss_ in epoch_loss:
            epoch_loss[loss_] = epoch_loss[loss_].cpu().item() / len(data_loader)

    return epoch_loss, val_freq_step


def train_mlpf(
    rank,
    deepmet,
    which_deepmet,
    mlpf,
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
    val_freq_step = 0
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        losses_t, val_freq_step = train_and_valid(
            rank,
            outdir,
            deepmet,
            which_deepmet,
            mlpf,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=True,
            epoch=epoch,
            dtype=dtype,
            val_freq=val_freq,
            val_freq_step=val_freq_step,
            tensorboard_writer=tensorboard_writer_train,
        )

        losses_v, _ = train_and_valid(
            rank,
            outdir,
            deepmet,
            which_deepmet,
            mlpf,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=False,
            epoch=epoch,
            val_freq=val_freq,
            val_freq_step=None,
            dtype=dtype,
        )

        extra_state = {"epoch": epoch}
        if losses_v["MET"] < best_val_loss:
            best_val_loss = losses_v["MET"]
            stale_epochs = 0
            torch.save(
                {"model_state_dict": get_model_state_dict(deepmet), "optimizer_state_dict": optimizer.state_dict()},
                f"{outdir}/best_weights.pth",
            )
            save_checkpoint(f"{outdir}/best_weights.pth", deepmet, optimizer, extra_state)
        else:
            stale_epochs += 1

        if checkpoint_freq and (epoch != 0) and (epoch % checkpoint_freq == 0):
            checkpoint_dir = Path(outdir) / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = "{}/checkpoint-{:02d}-{:.6f}.pth".format(checkpoint_dir, epoch, losses_v["MET"])
            save_checkpoint(checkpoint_path, deepmet, optimizer, extra_state)

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
