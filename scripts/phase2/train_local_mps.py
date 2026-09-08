#!/usr/bin/env python
"""Minimal local trainer (MPS/CPU) using the repo's MLPF model + loss on our
Run3-style Phase-2 pkls. Bypasses tfds/PFDataset (macOS array_record limitation)
but reuses mlpf.model.mlpf.MLPF, unpack_target/predictions, and mlpf_loss.

Live tracking: TensorBoard + an on-the-fly loss_curve.png + history.json + checkpoints,
all under --outdir, updated every epoch (safe to open any time)."""
import os, sys, glob, time, argparse, random, json
import numpy as np, torch
from types import SimpleNamespace
from collections import defaultdict
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "mlpf/heptfds/cms_pf_phase2"))
from mlpf.conf import MLPFConfig, Dataset, X_FEATURES, CLASS_LABELS, ELEM_TYPES_NONZERO
from mlpf.model.mlpf import MLPF
from mlpf.model.utils import unpack_target, unpack_predictions
from mlpf.model.losses import mlpf_loss

import cms_phase2_utils as adapter

REG_W = {"pt": 1.0, "eta": 0.01, "sin_phi": 0.01, "cos_phi": 0.01, "energy": 1.0}
NANO = os.path.join(REPO, "data/cms/phase2/offline/Aug31/nano")


def build_config(nc, nh, hd):
    ds = Dataset.CMS_PHASE2
    return MLPFConfig(
        dataset=ds, data_dir="/tmp/tfds_unused", conv_type="attention",
        input_dim=len(X_FEATURES[ds.value]), num_classes=len(CLASS_LABELS[ds.value]),
        elemtypes_nonzero=ELEM_TYPES_NONZERO[ds.value],
        model={"type": "attention", "input_encoding": "split",
               "attention": {"num_convs": nc, "head_dim": hd, "num_heads": nh, "attention_type": "math"}},
    )


def load_events(samples, max_files):
    evs = []
    for s in samples:
        fs = sorted(glob.glob(f"{NANO}/{s}/pkl_run3style/*.pkl"))
        if max_files > 0: fs = fs[:max_files]
        for f in fs:
            Xs, ytg, *_ = adapter.prepare_data_phase2(f)
            for X, y in zip(Xs, ytg):
                X = X.astype(np.float32); y = y.astype(np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    tp = np.log(y[:, 2] / X[:, 1]); te = np.log(y[:, 6] / X[:, 5])
                tp[~np.isfinite(tp)] = 0; te[~np.isfinite(te)] = 0; te[y[:, 0] == 0] = 0
                y[:, 2] = tp; y[:, 6] = te
                evs.append((X, y))
    random.Random(0).shuffle(evs)
    return evs


def make_batch(events, device):
    N = max(len(X) for X, _ in events); B = len(events)
    X = np.zeros((B, N, events[0][0].shape[1]), np.float32)
    Y = np.zeros((B, N, events[0][1].shape[1]), np.float32)
    mask = np.zeros((B, N), bool)
    for i, (x, y) in enumerate(events):
        X[i, :len(x)] = x; Y[i, :len(y)] = y; mask[i, :len(x)] = True
    return SimpleNamespace(X=torch.tensor(X, device=device), ytarget=torch.tensor(Y, device=device),
                           mask=torch.tensor(mask, device=device))


def plot_curves(hist, outdir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.plot(hist["epoch"], hist["train"], "o-", label="train")
    a1.plot(hist["epoch"], hist["val"], "s-", label="val")
    a1.set_xlabel("epoch"); a1.set_ylabel("total loss"); a1.set_title("Total loss"); a1.legend(); a1.grid(alpha=.3)
    for k in hist["comp_keys"]:
        a2.plot(hist["epoch"], hist[k], ".-", label=k.replace("Regression_", "reg:").replace("Classification", "cls"))
    a2.set_xlabel("epoch"); a2.set_ylabel("val loss component"); a2.set_yscale("log")
    a2.set_title("Loss components (val)"); a2.legend(fontsize=8); a2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=["ttbar_0pu", "qcd_0pu", "zll_0pu"])
    ap.add_argument("--max-files", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-convs", type=int, default=3)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--outdir", default=os.path.join(REPO, "experiments/phase2_local"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--class-weight-beta", type=float, default=0.5,
                    help="focal-alpha exponent on inverse class frequency (0=off, 0.5=sqrt-inv, 1=inv)")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True); os.makedirs(os.path.join(a.outdir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(a.outdir, "tb"))
    dev = a.device; print(f"device={dev} | outdir={a.outdir}", flush=True)

    cfg = build_config(a.num_convs, a.num_heads, a.head_dim)
    model = MLPF(cfg).to(dev)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"| input_dim={cfg.input_dim} num_classes={cfg.num_classes} convs={a.num_convs}", flush=True)

    t0 = time.time(); evs = load_events(a.samples, a.max_files)
    nval = max(1, len(evs) // 10); val, train = evs[:nval], evs[nval:]
    print(f"loaded {len(evs)} events in {time.time()-t0:.0f}s | train={len(train)} val={len(val)}", flush=True)

    # focal-alpha class weights: inverse class frequency ^ beta, normalized to mean 1 over present classes
    class_weights = None
    if a.class_weight_beta > 0:
        cnt = np.zeros(cfg.num_classes)
        for _, y in train:
            ids, c = np.unique(y[:, 0].astype(int), return_counts=True)
            cnt[ids] += c
        w = np.ones(cfg.num_classes)
        real = np.arange(cfg.num_classes) > 0
        inv = (1.0 / np.clip(cnt, 1, None)) ** a.class_weight_beta
        w[real] = inv[real] / inv[real].mean()      # normalize so mean weight over real classes = 1
        class_weights = torch.tensor(w, dtype=torch.float32, device=dev)
        names = ["null", "chad", "nhad", "photon", "elec", "muon"]
        print("class counts: " + "  ".join(f"{names[i]}={int(cnt[i])}" for i in range(cfg.num_classes)), flush=True)
        print("focal alpha (beta=%.2f): " % a.class_weight_beta
              + "  ".join(f"{names[i]}={w[i]:.2f}" for i in range(cfg.num_classes)), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    def run_epoch(data, training):
        model.train(training); tot = 0.0; comp = defaultdict(float); nb = 0
        for bi in range(0, len(data), a.batch_size):
            batch = make_batch(data[bi:bi+a.batch_size], dev)
            with torch.set_grad_enabled(training):
                ypred = unpack_predictions(model(batch.X, batch.mask))
                ytarget = unpack_target(batch.ytarget, model)
                loss_opt, losses = mlpf_loss(ytarget, ypred, batch, REG_W, class_weights)
            if training:
                opt.zero_grad(); loss_opt.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss_opt.item(); nb += 1
            for k, v in losses.items():
                comp[k] += float(v)
        return tot/max(nb, 1), {k: v/max(nb, 1) for k, v in comp.items()}

    hist = defaultdict(list); best = 1e9; best_ep = -1
    comp_keys = ["Classification_binary", "Classification", "Regression_pt", "Regression_energy",
                 "Regression_eta", "Regression_sin_phi", "Regression_cos_phi"]
    hist["comp_keys"] = comp_keys
    print(f"\n{'epoch':>5} {'train':>9} {'val':>9} {'sec':>6}", flush=True)
    for ep in range(a.epochs):
        t = time.time()
        tr, _ = run_epoch(train, True)
        vl, vcomp = run_epoch(val, False)
        dt = time.time()-t
        hist["epoch"].append(ep); hist["train"].append(tr); hist["val"].append(vl)
        for k in comp_keys: hist[k].append(vcomp.get(k, 0.0))
        writer.add_scalar("loss/train", tr, ep); writer.add_scalar("loss/val", vl, ep)
        for k in comp_keys: writer.add_scalar(f"val/{k}", vcomp.get(k, 0.0), ep)
        writer.flush()
        plot_curves(hist, a.outdir)
        json.dump({k: hist[k] for k in hist if k != "comp_keys"}, open(os.path.join(a.outdir, "history.json"), "w"))
        # full checkpoint (resumable): model + optimizer + config, EVERY epoch + best
        ckpt = {"epoch": ep, "train_loss": tr, "val_loss": vl,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                "args": vars(a), "config": {"num_convs": a.num_convs, "num_heads": a.num_heads, "head_dim": a.head_dim}}
        torch.save(ckpt, os.path.join(a.outdir, "checkpoints", f"checkpoint-{ep:02d}.pth"))
        torch.save(ckpt, os.path.join(a.outdir, "checkpoints", "last.pth"))
        if vl < best:
            best = vl; best_ep = ep
            torch.save(ckpt, os.path.join(a.outdir, "checkpoints", "best.pth"))
        json.dump({"best_epoch": best_ep, "best_val_loss": best, "last_epoch": ep,
                   "per_epoch_val": hist["val"]}, open(os.path.join(a.outdir, "best_info.json"), "w"), indent=2)
        print(f"{ep:>5} {tr:>9.4f} {vl:>9.4f} {dt:>6.1f}   (best: ep {best_ep}, val {best:.4f})", flush=True)
    print(f"\ndone. best val loss={best:.4f} at epoch {best_ep}", flush=True)


if __name__ == "__main__":
    main()
