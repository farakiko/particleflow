import pickle as pkl
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from pyg.logger import _logger
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
    deepmet,
    mlpf,
    use_latentX,
    optimizer,
    train_loader,
    valid_loader,
    trainable,
    is_train=True,
    epoch=None,
):
    """
    Performs training over a given epoch. Will run a validation step every val_freq.
    """

    train_or_valid = "train" if is_train else "valid"
    _logger.info(f"Initiating epoch #{epoch} {train_or_valid} run on device rank={rank}", color="red")

    configure_model_trainable(deepmet, trainable, is_train)

    if is_train:
        data_loader = train_loader
    else:
        data_loader = valid_loader

    # only show progress bar on rank 0
    iterator = tqdm.tqdm(
        enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} {train_or_valid} loop on rank={rank}"
    )

    epoch_loss = {}  # this one will keep accumulating `train_loss` and then return the average

    loss = {}  # this one is redefined every iteration

    for itrain, batch in iterator:

        batch = batch.to(rank, non_blocking=True)
        ygen = unpack_target(batch.ygen)

        if use_latentX:
            X = batch["mlpfcands_latentX"]
        else:
            X = torch.cat([batch["mlpfcands_momentum"], batch["mlpfcands_pid"], batch["mlpfcands_charge"]], axis=-1)

        assert X.requires_grad is False, "The MLPF model must be frozen."

        # define the model (JET model)
        if is_train:
            wx, wy = deepmet(X)
        else:
            with torch.no_grad():
                wx, wy = deepmet(X)

        # define the loss
        ptcorr = model(X, batch.batch).squeeze(1)

        target = torch.log(batch["gen_jet_pt"] / batch["reco_jet_pt"])

        if is_train:
            loss = torch.nn.functional.huber_loss(target, ptcorr)
            for param in deepmet.parameters():
                param.grad = None
            loss["MET"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = torch.nn.functional.huber_loss(target, ptcorr)

        # monitor the MLPF (and PF) MET loss
        with torch.no_grad():
            loss["MET_mlpf"] = torch.nn.functional.huber_loss(
                true_met_x, torch.sum(pred_px, axis=1)
            ) + torch.nn.functional.huber_loss(true_met_y, torch.sum(pred_py, axis=1))

            ycand = unpack_target(batch.ycand)
            msk_ycand = ycand["cls_id"] != 0
            cand_px = (ycand["pt"] * ycand["cos_phi"]) * msk_ycand
            cand_py = (ycand["pt"] * ycand["sin_phi"]) * msk_ycand

            loss["MET_PF"] = torch.nn.functional.huber_loss(
                true_met_x, torch.sum(cand_px, axis=1)
            ) + torch.nn.functional.huber_loss(true_met_y, torch.sum(cand_py, axis=1))

        for loss_ in loss.keys():
            if loss_ not in epoch_loss:
                epoch_loss[loss_] = 0.0
            epoch_loss[loss_] += loss[loss_].detach()

    for loss_ in epoch_loss:
        epoch_loss[loss_] = epoch_loss[loss_].cpu().item() / len(data_loader)

    return epoch_loss


def train_mlpf(
    rank,
    deepmet,
    mlpf,
    use_latentX,
    optimizer,
    train_loader,
    valid_loader,
    num_epochs,
    patience,
    outdir,
    trainable="all",
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

    losses_of_interest = ["MET", "MET_mlpf", "MET_PF"]

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
            deepmet,
            mlpf,
            use_latentX,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=True,
            epoch=epoch,
        )

        losses_v = train_and_valid(
            rank,
            deepmet,
            mlpf,
            use_latentX,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=False,
            epoch=epoch,
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

        for loss in losses_of_interest:
            losses["train"][loss].append(losses_t[loss])
            losses["valid"][loss].append(losses_v[loss])

        for k, v in losses_t.items():
            tensorboard_writer_train.add_scalar("epoch/loss_" + k, v, epoch)

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
