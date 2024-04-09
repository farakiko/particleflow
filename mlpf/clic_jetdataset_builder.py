import argparse
import os
import pickle as pkl
import random

import awkward as ak
import fastjet

# import relevant functions from mlpf.pyg
import torch
import tqdm
import vector
import yaml
from jet_utils import match_two_jet_collections
from pyg.mlpf import MLPF
from pyg.PFDataset import get_interleaved_dataloaders
from pyg.training_met import override_config
from pyg.utils import Y_FEATURES, load_checkpoint, unpack_predictions
from torch_geometric.data import Data


####################################
# must update this function to have the proper p4
def unpack_target(y):
    ret = {}
    ret["cls_id"] = y[..., 0].long()
    ret["charge"] = torch.clamp((y[..., 1] + 1).to(dtype=torch.float32), 0, 2)  # -1, 0, 1 -> 0, 1, 2

    for i, feat in enumerate(Y_FEATURES):
        if i >= 2:  # skip the cls and charge as they are defined above
            ret[feat] = y[..., i].to(dtype=torch.float32)
    ret["phi"] = torch.atan2(ret["sin_phi"], ret["cos_phi"])

    # do some sanity checks
    # assert torch.all(ret["pt"] >= 0.0)  # pt
    # assert torch.all(torch.abs(ret["sin_phi"]) <= 1.0)  # sin_phi
    # assert torch.all(torch.abs(ret["cos_phi"]) <= 1.0)  # cos_phi
    # assert torch.all(ret["energy"] >= 0.0)  # energy

    # note ~ momentum = ["pt", "eta", "sin_phi", "cos_phi", "energy"]
    ret["momentum"] = y[..., 2:7].to(dtype=torch.float32)
    ret["p4"] = torch.cat(
        [ret["pt"].unsqueeze(-1), ret["eta"].unsqueeze(-1), ret["phi"].unsqueeze(-1), ret["energy"].unsqueeze(-1)], axis=-1
    )

    ret["genjet_idx"] = y[..., -1].long()

    return ret


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

parser.add_argument("--save-every-X-batch", type=int, default=10, help="")


def main():
    args = parser.parse_args()

    with open(args.config, "r") as stream:  # load config (includes: which physics samples, model params)
        config = yaml.safe_load(stream)

    # override loaded config with values from command line args
    config = override_config(config, args)

    if config["gpus"]:
        assert torch.cuda.device_count() > 0, "--No gpu available"

        torch.cuda.empty_cache()

        rank = 0
        print(f"Will use single-gpu: {torch.cuda.get_device_name(rank)}")
    else:
        rank = "cpu"
        print("Will use cpu")

    loaddir = "/pfvol/experiments/MLPF_clic_A100_1gpu_pyg-clic_20240322_233518_004447"

    with open(f"{loaddir}/model_kwargs.pkl", "rb") as f:
        mlpf_kwargs = pkl.load(f)

    mlpf_kwargs["attention_type"] = "flash"

    mlpf = MLPF(**mlpf_kwargs).to(torch.device(rank))
    checkpoint = torch.load(f"{loaddir}/best_weights.pth", map_location=torch.device(rank))

    mlpf = load_checkpoint(checkpoint, mlpf)
    mlpf.eval()

    print(mlpf)

    loaders = get_interleaved_dataloaders(
        1,
        rank,
        config,
        use_cuda=rank != "cpu",
        # use_cuda=False,
        pad_3d=True,
        use_ray=False,
    )

    ###############################
    # Set up forward hooks to retrive the latent representations of MLPF
    latent_reps = {}

    def get_activations(name):
        def hook(mlpf, input, output):
            latent_reps[name] = output.detach()

        return hook

    mlpf.conv_reg[0].dropout.register_forward_hook(get_activations("conv_reg0"))
    mlpf.conv_reg[1].dropout.register_forward_hook(get_activations("conv_reg1"))
    mlpf.conv_reg[2].dropout.register_forward_hook(get_activations("conv_reg2"))
    mlpf.nn_id.register_forward_hook(get_activations("nn_id"))
    ###############################

    def get_latent_reps(batch, latent_reps):
        for layer in latent_reps:
            if "conv" in layer:
                latent_reps[layer] *= batch.mask.unsqueeze(-1)

        latentX = torch.cat(
            [
                batch.X.to(rank),
                latent_reps["conv_reg0"],
                latent_reps["conv_reg1"],
                latent_reps["conv_reg2"],
                latent_reps["nn_id"],
            ],
            axis=-1,
        )
        return latentX

    sample_to_lab = {
        "clic_edm_ttbar_pf": 1,
        "clic_edm_qq_pf": 0,
    }
    sample = "clic_edm_ttbar_pf"

    #######################
    # Config
    jetdef = fastjet.JetDefinition(fastjet.ee_genkt_algorithm, 0.7, -1.0)
    jet_ptcut = 15.0
    jet_match_dr = 0.1

    save_every_X_batch = args.save_every_X_batch  # will save to disk every "X" batches

    ########################
    # Build the dataset
    jet_dataset = []  # will save on disk and reinitialize at the end of the loop
    saving_i = 0  # will just increment with every save

    for mode in ["train", "test"]:

        outpath = f"/pfvol/jetdataset/{sample}/{mode}"
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        print(f"Will process the {mode} files")
        for ibatch, batch in enumerate(loaders[mode]):

            # run the MLPF model in inference mode to get the MLPF cands / latent representations
            print(f"Running MLPF inference on batch {ibatch}")
            batch = batch.to(rank, non_blocking=True)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    ymlpf = mlpf(batch.X, batch.mask)
            ymlpf = unpack_predictions(ymlpf)

            # get the latent representations
            ymlpf["latentX"] = get_latent_reps(batch, latent_reps)

            for k, v in ymlpf.items():
                ymlpf[k] = v.detach().cpu()

            msk_ymlpf = ymlpf["cls_id"] != 0
            ymlpf["p4"] = ymlpf["p4"] * msk_ymlpf.unsqueeze(-1)

            jets_coll = {}
            #######################
            # get the reco jet collection
            vec = vector.awk(
                ak.zip(
                    {
                        "pt": ymlpf["p4"][:, :, 0].to("cpu"),
                        "eta": ymlpf["p4"][:, :, 1].to("cpu"),
                        "phi": ymlpf["p4"][:, :, 2].to("cpu"),
                        "e": ymlpf["p4"][:, :, 3].to("cpu"),
                    }
                )
            )
            cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
            jets_coll["reco"] = cluster.inclusive_jets(min_pt=jet_ptcut)

            # get the constituents to mask the MLPF candidates and build the input for the downstream
            reco_constituents = cluster.constituent_index(min_pt=jet_ptcut)
            #######################

            #######################
            # get the gen jet collection
            ygen = unpack_target(batch.ygen)
            vec = vector.awk(
                ak.zip(
                    {
                        "pt": ygen["p4"][:, :, 0].to("cpu"),
                        "eta": ygen["p4"][:, :, 1].to("cpu"),
                        "phi": ygen["p4"][:, :, 2].to("cpu"),
                        "e": ygen["p4"][:, :, 3].to("cpu"),
                    }
                )
            )
            cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
            jets_coll["gen"] = cluster.inclusive_jets(min_pt=jet_ptcut)
            #######################

            matched_jets = match_two_jet_collections(jets_coll, "reco", "gen", jet_match_dr)

            # build the big jet list
            for iev in tqdm.tqdm(range(len(matched_jets["gen"]))):

                num_matched_jets = len(matched_jets["gen"][iev])  # number of gen jets matched to reco

                jets_per_event = []
                for j in range(num_matched_jets):

                    # get the actual indices of the matched jets
                    igenjet = matched_jets["gen"][iev][j]
                    irecojet = matched_jets["reco"][iev][j]

                    # build a mask tensor that will select the particles that belong to the gen jet
                    msk_indices = reco_constituents[iev][irecojet].to_numpy()

                    if len(msk_indices) < 3:
                        # don't save jets with very few particles
                        continue

                    jets_per_event += [
                        Data(
                            # Target for jet tagging
                            gen_jet_label=torch.tensor(sample_to_lab[sample]).unsqueeze(0).to(dtype=torch.float32),
                            # Target for jet p4 regression
                            gen_jet_pt=torch.tensor(jets_coll["gen"][iev][igenjet].pt, dtype=torch.float32).unsqueeze(0),
                            gen_jet_eta=torch.tensor(jets_coll["gen"][iev][igenjet].eta, dtype=torch.float32).unsqueeze(0),
                            gen_jet_phi=torch.tensor(jets_coll["gen"][iev][igenjet].phi, dtype=torch.float32).unsqueeze(0),
                            gen_jet_energy=torch.tensor(
                                jets_coll["gen"][iev][igenjet].energy, dtype=torch.float32
                            ).unsqueeze(0),
                            # could be part of the target
                            reco_jet_pt=torch.tensor(jets_coll["reco"][iev][irecojet].pt, dtype=torch.float32).unsqueeze(0),
                            reco_jet_eta=torch.tensor(jets_coll["reco"][iev][irecojet].eta, dtype=torch.float32).unsqueeze(
                                0
                            ),
                            reco_jet_phi=torch.tensor(jets_coll["reco"][iev][irecojet].phi, dtype=torch.float32).unsqueeze(
                                0
                            ),
                            reco_jet_energy=torch.tensor(
                                jets_coll["reco"][iev][irecojet].energy, dtype=torch.float32
                            ).unsqueeze(0),
                            # Input
                            mlpfcands_momentum=ymlpf["momentum"][iev][msk_indices],
                            mlpfcands_pid=ymlpf["cls_id_onehot"][iev][msk_indices],
                            mlpfcands_charge=ymlpf["charge"][iev][msk_indices],
                            mlpfcands_latentX=ymlpf["latentX"][iev][msk_indices],
                        )
                    ]

                #             break  # per jet

                random.shuffle(jets_per_event)
                jet_dataset += jets_per_event

            #         break   # per event

            random.shuffle(jet_dataset)
            if (ibatch % (save_every_X_batch - 1) == 0) and (ibatch != 0):
                print(f"saving at iteration {ibatch} on disk {outpath}/{saving_i}.pt")
                torch.save(jet_dataset, f"{outpath}/{saving_i}.pt")
                saving_i += 1
                jet_dataset = []
                # break  # per batch


if __name__ == "__main__":
    # noqa: python mlpf/clic_jetdataset_builder.py --dataset clic --data-dir tensorflow_datasets --config parameters/pytorch/pyg-clic-ttbar.yaml --num-workers 4 --prefetch-factor 20 --gpu-batch-multiplier 100 --save-every-X-batch 2
    main()
