"""
Developing a PyTorch Geometric supervised training of MLPF using DistributedDataParallel.

Authors: Farouk Mokhtar, Joosep Pata, Eric Wulff
"""

import argparse

import torch
import tqdm
import yaml
from pyg.PFDataset import get_interleaved_dataloaders
from pyg.training_met import override_config

parser = argparse.ArgumentParser()
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


def main():
    args = parser.parse_args()

    with open(args.config, "r") as stream:  # load config (includes: which physics samples, model params)
        config = yaml.safe_load(stream)

    # override loaded config with values from command line args
    config = override_config(config, args)

    rank = "cpu"
    pad_3d = True
    use_cuda = False

    loaders = get_interleaved_dataloaders(
        1,
        rank,
        config,
        use_cuda,
        pad_3d,
        use_ray=False,
    )

    train_loader = []
    print(f"train_loader: {len(loaders['train'])}")
    for i, batch in tqdm.tqdm(enumerate(loaders["train"])):
        train_loader += [batch]
        if i % 300 == 0:  # every 300 batches will save to disk
            torch.save(train_loader, f"/pfvol/torchdata/train/train_list_{i}.pt")
            print(f"saved /pfvol/torchdata/train/train_list_{i}.pt")
            train_loader = []

    valid_loader = []
    print(f"valid_loader: {len(loaders['valid'])}")
    for i, batch in tqdm.tqdm(enumerate(loaders["valid"])):
        valid_loader += [batch]
        if i % 300 == 0:  # every 300 batches will save to disk
            torch.save(valid_loader, f"/pfvol/torchdata/valid/valid_list_{i}.pt")
            print(f"saved /pfvol/torchdata/valid/valid_list_{i}.pt")
            valid_loader = []


if __name__ == "__main__":
    # noqa: python mlpf/clic_data.py --dataset clic --data-dir tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml --num-workers 4 --prefetch-factor 20 --gpu-batch-multiplier 100
    main()
