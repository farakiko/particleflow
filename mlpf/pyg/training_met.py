import pickle as pkl
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from pyg.logger import _logger
from pyg.training import FocalLoss
from pyg.utils import (
    get_model_state_dict,
    save_checkpoint,
    unpack_predictions,
    unpack_target,
)
from torch.utils.tensorboard import SummaryWriter

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


def mlpf_loss(y, ypred, batchidx_or_mask):
    """
    Args
        y [dict]: relevant keys are "cls_id, momentum, charge"
        ypred [dict]: relevant keys are "cls_id_onehot, momentum, charge"
    """
    loss = {}
    loss_obj_id = FocalLoss(gamma=2.0, reduction="none")

    msk_true_particle = torch.unsqueeze((y["cls_id"] != 0).to(dtype=torch.float32), axis=-1)

    nelem = torch.sum(batchidx_or_mask)

    npart = torch.sum(y["cls_id"] != 0)

    ypred["momentum"] = ypred["momentum"] * msk_true_particle
    # ypred["charge"] = ypred["charge"] * msk_true_particle
    y["momentum"] = y["momentum"] * msk_true_particle
    # y["charge"] = y["charge"] * msk_true_particle[..., 0]

    # in case of the 3D-padded mode, pytorch expects (N, C, ...)
    ypred["cls_id_onehot"] = ypred["cls_id_onehot"].permute((0, 2, 1))
    # ypred["charge"] = ypred["charge"].permute((0, 2, 1))

    loss_classification = 100 * loss_obj_id(ypred["cls_id_onehot"], y["cls_id"]).reshape(y["cls_id"].shape)
    loss_regression = 10 * torch.nn.functional.huber_loss(ypred["momentum"], y["momentum"], reduction="none")
    # loss_charge = 0.0*torch.nn.functional.cross_entropy(
    #     ypred["charge"], y["charge"].to(dtype=torch.int64), reduction="none")

    # average over all elements that were not padded
    loss["Classification"] = loss_classification.sum() / nelem

    # normalize loss with stddev to stabilize across batches with very different pt, E distributions
    mom_normalizer = y["momentum"][y["cls_id"] != 0].std(axis=0)
    reg_losses = loss_regression[y["cls_id"] != 0]
    # average over all true particles
    loss["Regression"] = (reg_losses / mom_normalizer).sum() / npart
    # loss["Charge"] = loss_charge.sum() / npart

    # we can compute a few additional event-level monitoring losses
    msk_pred_particle = torch.unsqueeze(torch.argmax(ypred["cls_id_onehot"].detach(), axis=1) != 0, axis=-1)

    px = ypred["momentum"][..., 0:1] * ypred["momentum"][..., 3:4] * msk_pred_particle  # pt * cos_phi
    py = ypred["momentum"][..., 0:1] * ypred["momentum"][..., 2:3] * msk_pred_particle  # pt * sin_phi
    pred_met = torch.sum(px, axis=-2) ** 2 + torch.sum(py, axis=-2) ** 2

    px = y["momentum"][..., 0:1] * y["momentum"][..., 3:4] * msk_true_particle
    py = y["momentum"][..., 0:1] * y["momentum"][..., 2:3] * msk_true_particle
    true_met = torch.sum(px, axis=-2) ** 2 + torch.sum(py, axis=-2) ** 2

    loss["MET"] = torch.nn.functional.huber_loss(pred_met, true_met).detach().mean()

    loss["Total"] = loss["Classification"] + loss["Regression"]  # + loss["Charge"]

    loss["Classification"] = loss["Classification"].detach()
    loss["Regression"] = loss["Regression"].detach()
    # loss["Charge"] = loss["Charge"].detach()
    # print(loss["Total"].detach().item(), y["cls_id"].shape, nelem, npart)
    return loss


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
    model,
    optimizer,
    train_loader,
    valid_loader,
    trainable,
    is_train=True,
    lr_schedule=None,
    epoch=None,
    dtype=torch.float32,
    tensorboard_writer=None,
):
    """
    Performs training over a given epoch. Will run a validation step every N_STEPS and after the last training batch.
    """

    train_or_valid = "train" if is_train else "valid"
    _logger.info(f"Initiating epoch #{epoch} {train_or_valid} run on device rank={rank}", color="red")

    # this one will keep accumulating `train_loss` and then return the average
    epoch_loss = {}

    configure_model_trainable(model, trainable, is_train)
    if is_train:
        data_loader = train_loader
    else:
        data_loader = valid_loader

    # only show progress bar on rank 0
    iterator = tqdm.tqdm(
        enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} {train_or_valid} loop on rank={rank}"
    )

    device_type = "cuda" if isinstance(rank, int) else "cpu"

    loss_accum = 0.0
    for itrain, batch in iterator:
        batch = batch.to(rank, non_blocking=True)

        ygen = unpack_target(batch.ygen)

        batchidx_or_mask = batch.mask
        num_elems = batch.X[batch.mask].shape[0]
        num_batch = batch.X.shape[0]

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
            if is_train:
                ypred = model(batch.X, batchidx_or_mask)
            else:
                with torch.no_grad():
                    ypred = model(batch.X, batchidx_or_mask)

        ypred = unpack_predictions(ypred)

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
            if is_train:
                loss = mlpf_loss(ygen, ypred, batchidx_or_mask)
                for param in model.parameters():
                    param.grad = None
            else:
                with torch.no_grad():
                    loss = mlpf_loss(ygen, ypred, batchidx_or_mask)

        if is_train:
            loss["Total"].backward()
            loss_accum += loss["Total"].detach().cpu().item()
            optimizer.step()
            if lr_schedule:
                lr_schedule.step()

        for loss_ in loss.keys():
            if loss_ not in epoch_loss:
                epoch_loss[loss_] = 0.0
            epoch_loss[loss_] += loss[loss_].detach()

        if is_train:
            step = (epoch - 1) * len(data_loader) + itrain
            if not (tensorboard_writer is None):
                tensorboard_writer.add_scalar("step/loss", loss_accum / num_elems, step)
                tensorboard_writer.add_scalar("step/num_elems", num_elems, step)
                tensorboard_writer.add_scalar("step/num_batch", num_batch, step)
                tensorboard_writer.add_scalar("step/learning_rate", lr_schedule.get_last_lr()[0], step)
                if itrain % 10 == 0:
                    tensorboard_writer.flush()
                loss_accum = 0.0

    num_data = torch.tensor(len(data_loader), device=rank)
    # sum up the number of steps from all workers

    for loss_ in epoch_loss:
        # sum up the losses from all workers
        epoch_loss[loss_] = epoch_loss[loss_].cpu().item() / num_data.cpu().item()

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
    lr_schedule=None,
    checkpoint_freq=None,
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

    tensorboard_writer_train = SummaryWriter(f"{outdir}/runs/train")
    tensorboard_writer_valid = SummaryWriter(f"{outdir}/runs/valid")

    t0_initial = time.time()

    losses_of_interest = ["Total", "Classification", "Regression"]

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
            model,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=True,
            lr_schedule=lr_schedule,
            epoch=epoch,
            dtype=dtype,
            tensorboard_writer=tensorboard_writer_train,
        )

        losses_v = train_and_valid(
            rank,
            model,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=False,
            lr_schedule=None,
            epoch=epoch,
            dtype=dtype,
        )

        tensorboard_writer_train.add_scalar("epoch/learning_rate", lr_schedule.get_last_lr()[0], epoch)

        extra_state = {"epoch": epoch, "lr_schedule_state_dict": lr_schedule.state_dict()}
        if losses_v["Total"] < best_val_loss:
            best_val_loss = losses_v["Total"]
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
            checkpoint_path = "{}/checkpoint-{:02d}-{:.6f}.pth".format(checkpoint_dir, epoch, losses_v["Total"])
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
            + f"train_loss={losses_t['Total']:.4f} "
            + f"valid_loss={losses_v['Total']:.4f} "
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
