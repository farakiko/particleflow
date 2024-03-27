"""
Developing a PyTorch Geometric supervised training of MLPF using DistributedDataParallel.

Authors: Farouk Mokhtar, Joosep Pata, Eric Wulff
"""

import argparse
import logging
import pickle as pkl
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from pyg.logger import _configLogger, _logger
from pyg.mlpf import MLPF
from pyg.PFDataset import get_interleaved_dataloaders
from pyg.training_met import override_config, train_mlpf
from pyg.utils import load_checkpoint, save_HPs
from utils import create_experiment_dir

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

# add default=None to all arparse arguments to ensure they do not override
# values loaded from the config file given by --config unless explicitly given
parser.add_argument("--config", type=str, default=None, help="yaml config")
parser.add_argument("--prefix", type=str, default=None, help="prefix appended to result dir name")
parser.add_argument("--data-dir", type=str, default=None, help="path to `tensorflow_datasets/`")
parser.add_argument("--gpus", type=int, default=None, help="to use CPU set to 0; else e.g., 4")
parser.add_argument(
    "--gpu-batch-multiplier", type=int, default=None, help="Increase batch size per GPU by this constant factor"
)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    choices=["clic", "cms", "delphes", "clic_hits"],
    required=False,
    help="which dataset?",
)
parser.add_argument("--num-workers", type=int, default=None, help="number of processes to load the data")
parser.add_argument("--prefetch-factor", type=int, default=None, help="number of samples to fetch & prefetch at every call")
parser.add_argument("--load", type=str, default=None, help="load checkpoint and start new training from epoch 1")
parser.add_argument("--train", action="store_true", default=None, help="initiates a training")
parser.add_argument("--test", action="store_true", default=None, help="tests the model")
parser.add_argument("--num-epochs", type=int, default=None, help="number of training epochs")
parser.add_argument("--patience", type=int, default=None, help="patience before early stopping")
parser.add_argument("--lr", type=float, default=None, help="learning rate")
parser.add_argument(
    "--conv-type",
    type=str,
    default=None,
    help="which graph layer to use",
    choices=["gravnet", "attention", "gnn_lsh", "mamba"],
)
parser.add_argument("--num-convs", type=int, default=None, help="number of convlution (GNN, attention, Mamba) layers")
parser.add_argument("--make-plots", action="store_true", default=None, help="make plots of the test predictions")
parser.add_argument("--export-onnx", action="store_true", default=None, help="exports the model to onnx")
parser.add_argument("--ntrain", type=int, default=None, help="training samples to use, if None use entire dataset")
parser.add_argument("--ntest", type=int, default=None, help="training samples to use, if None use entire dataset")
parser.add_argument("--nvalid", type=int, default=None, help="validation samples to use")
parser.add_argument("--val-freq", type=int, default=None, help="run extra validation every val_freq training steps")
parser.add_argument("--checkpoint-freq", type=int, default=None, help="epoch frequency for checkpointing")
parser.add_argument("--in-memory", action="store_true", default=None, help="if True will load the data into memory first")
parser.add_argument("--numtrain", type=int, default=10000, help="training samples to use")
parser.add_argument("--numvalid", type=int, default=1000, help="validation samples to use")

parser.add_argument(
    "--dtype",
    type=str,
    default=None,
    help="data type for training",
    choices=["float32", "float16", "bfloat16"],
)
parser.add_argument(
    "--attention-type",
    type=str,
    default=None,
    help="attention type for self-attention layer",
    choices=["math", "efficient", "flash", "flash_external"],
)

parser.add_argument(
    "--which-deepmet",
    type=str,
    default="1",
    help="which deepmet to use",
    choices=["1", "2"],
)

parser.add_argument("--use-PFcands", action="store_true", default=None, help="if True will not make use of MLPF")


class DeepMET1(nn.Module):
    def __init__(
        self,
        width=128,
    ):
        super(DeepMET1, self).__init__()

        """
        Takes as input the p4 of the MLPF/PF candidates; will run an encoder -> decoder to learn
        two outputs per particle "w_xi" and "w_yi" which will enter the loss:
            MET^2 = (sum_(w_x * pxi)^2) + sum_(w_y * pxi)^2)
        """

        self.act = nn.ELU

        regression_nodes = 5
        self.input_dim = regression_nodes

        self.nn_encoder = nn.Sequential(
            nn.Linear(self.input_dim, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
        )

        self.nn_decoder = nn.Sequential(
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, 2),
        )

    # @torch.compile
    def forward(self, X):

        encoded_element = self.nn_encoder(X)

        MET = self.nn_decoder(encoded_element)

        return MET[:, :, 0], MET[:, :, 1]


class DeepMET2(nn.Module):
    def __init__(
        self,
        width=128,
    ):
        super(DeepMET2, self).__init__()

        """
        Takes as input the p4 of the MLPF/PF candidates; will run an encoder -> pooling -> decoder to learn
        two outputs per event "w_x" and "w_y" which will enter the loss:
            MET^2 = (w_x * (sum_pxi)^2) + (w_y * (sum_pyi)^2)
        """

        self.act = nn.ELU

        regression_nodes = 5
        self.input_dim = regression_nodes

        self.nn_encoder = nn.Sequential(
            nn.Linear(self.input_dim, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
        )

        self.nn_decoder = nn.Sequential(
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, 2),
        )

    # @torch.compile
    def forward(self, X):

        encoded_element = self.nn_encoder(X)

        encoded_element = encoded_element.sum(axis=1)  # pool over particles; recall ~ [Batch, Particles, Feature]

        MET = self.nn_decoder(encoded_element)

        return MET[:, 0], MET[:, 1]


def main():
    args = parser.parse_args()

    with open(args.config, "r") as stream:  # load config (includes: which physics samples, model params)
        config = yaml.safe_load(stream)

    # override loaded config with values from command line args
    config = override_config(config, args)

    assert config["load"], "Must pass an MLPF model to --load"

    if "best_weights" in Path(config["load"]).name:
        loaddir = str(Path(config["load"]).parent)
    else:
        # the checkpoint is provided directly
        loaddir = str(Path(config["load"]).parent.parent)

    if args.use_PFcands:
        append_ = "PFcands"
    else:
        append_ = "MLPFcands"

    outdir = create_experiment_dir(
        prefix=(args.prefix or "") + f"_{append_}_DeepMET{args.which_deepmet}_",
        experiments_dir=loaddir,
    )

    # Save config for later reference. Note that saving happens after parameters are overwritten by cmd line args.
    config_filename = "train-config.yaml" if args.train else "test-config.yaml"
    with open((Path(outdir) / config_filename), "w") as file:
        yaml.dump(config, file)

    if args.train:
        logfile = f"{outdir}/train.log"
        _configLogger("mlpf", filename=logfile)
    else:
        outdir = str(Path(args.load).parent.parent)
        logfile = f"{outdir}/test.log"
        _configLogger("mlpf", filename=logfile)

    if config["gpus"]:
        assert torch.cuda.device_count() > 0, "--No gpu available"

        torch.cuda.empty_cache()

        rank = 0
        _logger.info(f"Will use single-gpu: {torch.cuda.get_device_name(rank)}", color="purple")

    else:
        rank = "cpu"
        _logger.info("Will use cpu", color="purple")

    pad_3d = True
    use_cuda = rank != "cpu"

    dtype = getattr(torch, config["dtype"])
    _logger.info("using dtype={}".format(dtype))

    _configLogger("mlpf", filename=logfile)

    # load the mlpf model
    if args.use_PFcands:
        _logger.info("Will use the PF candidates as input so no need to load MLPF", color="orange")

        mlpf = {}
        mlpf_kwargs = {}

    else:
        _logger.info("Will use the MLPF cands", color="orange")

        with open(f"{loaddir}/model_kwargs.pkl", "rb") as f:
            mlpf_kwargs = pkl.load(f)
        _logger.info("mlpf_kwargs: {}".format(mlpf_kwargs))

        mlpf_kwargs["attention_type"] = config["model"]["attention"]["attention_type"]

        mlpf = MLPF(**mlpf_kwargs).to(torch.device(rank))
        checkpoint = torch.load(config["load"], map_location=torch.device(rank))

        mlpf = load_checkpoint(checkpoint, mlpf)
        mlpf.eval()

        _logger.info(mlpf)

    # define the deepmet model
    if args.which_deepmet == "1":
        _logger.info("Will use DeepMET model #1 (without pooling)", color="orange")

        deepmet = DeepMET1().to(torch.device(rank))
    else:
        _logger.info("Will use DeepMET model #2 (with pooling)", color="orange")

        deepmet = DeepMET2().to(torch.device(rank))

    deepmet.train()
    optimizer = torch.optim.AdamW(deepmet.parameters(), lr=args.lr)
    _logger.info(deepmet)

    if args.train:
        save_HPs(args, deepmet, mlpf_kwargs, outdir)  # save model_kwargs and hyperparameters
        _logger.info("Creating experiment dir {}".format(outdir))
        _logger.info(f"Model directory {outdir}", color="bold")

        loaders = get_interleaved_dataloaders(
            1,
            rank,
            config,
            use_cuda,
            pad_3d,
            use_ray=False,
        )

        # if args.in_memory:
        #     import tqdm
        #     train_loader = []
        #     for i, batch in tqdm.tqdm(enumerate(loaders["train"])):
        #         train_loader += [batch]
        #         if i == args.numtrain:
        #             break
        #     loaders["train"] = train_loader

        #     valid_loader = []
        #     for i, batch in tqdm.tqdm(enumerate(loaders["valid"])):
        #         valid_loader += [batch]
        #         if i == args.numvalid:
        #             break
        #     loaders["valid"] = valid_loader

        train_mlpf(
            rank,
            deepmet,
            args.which_deepmet,
            mlpf,
            optimizer,
            loaders["train"],
            loaders["valid"],
            config["num_epochs"],
            config["patience"],
            outdir,
            trainable=config["model"]["trainable"],
            dtype=dtype,
            checkpoint_freq=config["checkpoint_freq"],
            val_freq=config["val_freq"],
        )


if __name__ == "__main__":

    # e.g.
    # noqa: python mlpf/met_finetuning_pipeline.py --dataset clic --data-dir tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml --gpus 1 --prefix MLPF_test1 --num-epochs 10 --train --load /pfvol/experiments/MLPF_clic_A100_1gpu_pyg-clic_20240322_233518_004447/best_weights.pth --gpu-batch-multiplier 100 --num-workers 4 --prefetch-factor 20 --checkpoint-freq 1 --val-freq 1000
    # git checkout 8d9065cba1af49b97c63c4701789a7f7a1fbcd47 /home/jovyan/particleflow/mlpf/pyg/mlpf.py
    # git checkout 8d9065cba1af49b97c63c4701789a7f7a1fbcd47 /home/jovyan/particleflow/mlpf/pyg/utils.py
    main()
