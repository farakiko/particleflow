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
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    choices=["clic", "cms", "delphes", "clic_hits"],
    required=False,
    help="which dataset?",
)
parser.add_argument("--data-dir", type=str, default=None, help="path to `tensorflow_datasets/`")
parser.add_argument("--num-workers", type=int, default=None, help="number of processes to load the data")
parser.add_argument("--prefetch-factor", type=int, default=None, help="number of samples to fetch & prefetch at every call")
parser.add_argument(
    "--gpu-batch-multiplier", type=int, default=None, help="Increase batch size per GPU by this constant factor"
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

    j = 0
    train_loader = []
    for i, batch in tqdm.tqdm(enumerate(loaders["train"])):
        train_loader += [batch]
        if i % 300 == 0:  # every 300 batches will save to disk
            torch.save(train_loader, f"/pfvol/torchdata/train/train_list_{j}")
            j += 1
            train_loader = []

    j = 0
    valid_loader = []
    for i, batch in tqdm.tqdm(enumerate(loaders["valid"])):
        valid_loader += [batch]
        if i % 300 == 0:  # every 300 batches will save to disk
            torch.save(valid_loader, f"/pfvol/torchdata/valid/valid_list_{j}")
            j += 1
            valid_loader = []


if __name__ == "__main__":
    # noqa: python mlpf/clic_data.py --dataset clic --data-dir tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml --num-workers 4 --prefetch-factor 20
    main()
