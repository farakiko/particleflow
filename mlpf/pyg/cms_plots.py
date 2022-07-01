from pyg.cms_utils import CLASS_LABELS_CMS, CLASS_NAMES_CMS, CLASS_NAMES_CMS_LATEX

import pickle
import bz2

import numpy as np
from numpy.lib.recfunctions import append_fields

import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Dataset, Data, Batch

import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

import sklearn
import sklearn.metrics
import matplotlib
import scipy
import mplhep

import pandas
import itertools
import mplhep
import boost_histogram as bh
mplhep.style.use(mplhep.styles.CMS)


def cms_label(ax, x0=0.01, x1=0.15, x2=0.98, y=0.94):
    plt.figtext(x0, y, 'CMS', fontweight='bold', wrap=True, horizontalalignment='left', transform=ax.transAxes)
    plt.figtext(x1, y, 'Simulation Preliminary', style='italic', wrap=True, horizontalalignment='left', transform=ax.transAxes)
    plt.figtext(x2, y, 'Run 3 (14 TeV)',  wrap=False, horizontalalignment='right', transform=ax.transAxes)

# def cms_label_sample_label(x0=0.12, x1=0.23, x2=0.67, y=0.90):
#     plt.figtext(x0, y,'CMS',fontweight='bold', wrap=True, horizontalalignment='left')
#     plt.figtext(x1, y,'Simulation Preliminary', style='italic', wrap=True, horizontalalignment='left')
#     plt.figtext(x2, y,'Run 3 (14 TeV), $\mathrm{t}\overline{\mathrm{t}}$ events',  wrap=False, horizontalalignment='left')


def sample_label(sample, ax, additional_text="", x=0.01, y=0.87):
    if sample == 'QCD':
        plt.text(x, y, "QCD events" + additional_text, ha="left", transform=ax.transAxes)
    else:
        plt.text(x, y, "$\mathrm{t}\overline{\mathrm{t}}$ events" + additional_text, ha="left", transform=ax.transAxes)


def apply_thresholds_f(ypred_raw_f, thresholds):
    msk = np.ones_like(ypred_raw_f)
    for i in range(len(thresholds)):
        msk[:, i + 1] = ypred_raw_f[:, i + 1] > thresholds[i]
    ypred_id_f = np.argmax(ypred_raw_f * msk, axis=-1)

#     best_2 = np.partition(ypred_raw_f, -2, axis=-1)[..., -2:]
#     diff = np.abs(best_2[:, -1] - best_2[:, -2])
#     ypred_id_f[diff<0.05] = 0

    return ypred_id_f


def apply_thresholds(ypred_raw, thresholds):
    msk = np.ones_like(ypred_raw)
    for i in range(len(thresholds)):
        msk[:, :, i + 1] = ypred_raw[:, :, i + 1] > thresholds[i]
    ypred_id = np.argmax(ypred_raw * msk, axis=-1)

#     best_2 = np.partition(ypred_raw, -2, axis=-1)[..., -2:]
#     diff = np.abs(best_2[:, :, -1] - best_2[:, :, -2])
#     ypred_id[diff<0.05] = 0

    return ypred_id


class_names = {k: v for k, v in zip(CLASS_LABELS_CMS, CLASS_NAMES_CMS)}


def plot_numPFelements(X, outpath, sample):
    plt.figure()
    ax = plt.axes()
    plt.hist(np.sum(X[:, :, 0] != 0, axis=1), bins=100)
    plt.axvline(6400, ls="--", color="black")
    plt.xlabel("number of input PFElements")
    plt.ylabel("number of events / bin")
    cms_label(ax)
    sample_label(sample, ax)
    plt.savefig(f"{outpath}/num_PFelements.pdf", bbox_inches="tight")
    plt.close()


def plot_met(X, yvals, outpath, sample):
    sum_px = np.sum(yvals["gen_px"], axis=1)
    sum_py = np.sum(yvals["gen_py"], axis=1)
    gen_met = np.sqrt(sum_px**2 + sum_py**2)[:, 0]

    sum_px = np.sum(yvals["cand_px"], axis=1)
    sum_py = np.sum(yvals["cand_py"], axis=1)
    cand_met = np.sqrt(sum_px**2 + sum_py**2)[:, 0]

    sum_px = np.sum(yvals["pred_px"], axis=1)
    sum_py = np.sum(yvals["pred_py"], axis=1)
    pred_met = np.sqrt(sum_px**2 + sum_py**2)[:, 0]

    fig = plt.figure()
    ax = plt.axes()
    b = np.linspace(-2, 5, 101)
    vals_a = (cand_met - gen_met) / gen_met
    vals_b = (pred_met - gen_met) / gen_met
    plt.hist(vals_a, bins=b, histtype="step", lw=2, label="PF, $\mu={:.2f}$, $\sigma={:.2f}$".format(np.mean(vals_a), np.std(vals_a)))
    plt.hist(vals_b, bins=b, histtype="step", lw=2, label="MLPF, $\mu={:.2f}$, $\sigma={:.2f}$".format(np.mean(vals_b), np.std(vals_b)))
    plt.yscale("log")
    cms_label(ax)
    sample_label(sample, ax)
    plt.ylim(10, 1e3)
    plt.legend(loc=(0.4, 0.7))
    plt.xlabel(r"$\frac{\mathrm{MET}_{\mathrm{reco}} - \mathrm{MET}_{\mathrm{gen}}}{\mathrm{MET}_{\mathrm{gen}}}$")
    plt.ylabel("Number of events / bin")
    plt.savefig(f"{outpath}/met.pdf", bbox_inches="tight")
    plt.close()


def plot_sum_energy(X, yvals, outpath, sample):
    fig = plt.figure()
    ax = plt.axes()

    plt.scatter(
        np.sum(yvals["gen_energy"], axis=1),
        np.sum(yvals["cand_energy"], axis=1),
        alpha=0.5,
        label="PF"
    )
    plt.scatter(
        np.sum(yvals["gen_energy"], axis=1),
        np.sum(yvals["pred_energy"], axis=1),
        alpha=0.5,
        label="MLPF"
    )
    plt.plot([10000, 80000], [10000, 80000], color="black")
    plt.legend(loc=4)
    cms_label(ax)
    sample_label(sample, ax)
    plt.xlabel("Gen $\sum E$ [GeV]")
    plt.ylabel("Reconstructed $\sum E$ [GeV]")
    plt.savefig(f"{outpath}/sum_energy.pdf", bbox_inches="tight")
    plt.close()


def plot_sum_pt(X, yvals, outpath, sample):

    fig = plt.figure()
    ax = plt.axes()

    plt.scatter(
        np.sum(yvals["gen_pt"], axis=1),
        np.sum(yvals["cand_pt"], axis=1),
        alpha=0.5,
        label="PF"
    )
    plt.scatter(
        np.sum(yvals["gen_pt"], axis=1),
        np.sum(yvals["pred_pt"], axis=1),
        alpha=0.5,
        label="PF"
    )
    plt.plot([1000, 6000], [1000, 6000], color="black")
    plt.legend(loc=4)
    cms_label(ax)
    sample_label(sample, ax)
    plt.xlabel("Gen $\sum p_T$ [GeV]")
    plt.ylabel("Reconstructed $\sum p_T$ [GeV]")
    plt.savefig(f"{outpath}/sum_pt.pdf", bbox_inches="tight")
    plt.close()


def plot_energy_res(X, yvals_f, pid, b, ylim, outpath, sample):

    fig = plt.figure()
    ax = plt.axes()

    msk = (yvals_f["gen_cls_id"] == pid) & (yvals_f["cand_cls_id"] == pid) & (yvals_f["pred_cls_id"] == pid)
    vals_gen = yvals_f["gen_energy"][msk]
    vals_cand = yvals_f["cand_energy"][msk]
    vals_mlpf = yvals_f["pred_energy"][msk]

    reso_1 = (vals_cand - vals_gen) / vals_gen
    reso_2 = (vals_mlpf - vals_gen) / vals_gen
    plt.hist(reso_1, bins=b, histtype="step", lw=2, label="PF, $\mu={:.2f}, \sigma={:.2f}$".format(np.mean(reso_1), np.std(reso_1)))
    plt.hist(reso_2, bins=b, histtype="step", lw=2, label="MLPF, $\mu={:.2f}, \sigma={:.2f}$".format(np.mean(reso_2), np.std(reso_2)))
    plt.yscale("log")
    plt.xlabel(r"$\frac{E_\mathrm{reco} - E_\mathrm{gen}}{E_\mathrm{gen}}$")
    plt.ylabel("Number of particles / bin")
    cms_label(ax)
    sample_label(sample, ax, f", {CLASS_NAMES_CMS_LATEX[pid]}")
    plt.legend(loc=(0.4, 0.7))
    plt.ylim(1, ylim)
    plt.savefig(f"{outpath}/energy_res_{CLASS_NAMES_CMS[pid]}.pdf", bbox_inches="tight")
    plt.close()


def plot_eta_res(X, yvals_f, pid, ylim, outpath, sample):

    fig = plt.figure()
    ax = plt.axes()

    msk = (yvals_f["gen_cls_id"] == pid) & (yvals_f["cand_cls_id"] == pid) & (yvals_f["pred_cls_id"] == pid)
    vals_gen = yvals_f["gen_eta"][msk]
    vals_cand = yvals_f["cand_eta"][msk]
    vals_mlpf = yvals_f["pred_eta"][msk]

    b = np.linspace(-10, 10, 100)

    reso_1 = (vals_cand - vals_gen)
    reso_2 = (vals_mlpf - vals_gen)
    plt.hist(reso_1, bins=b, histtype="step", lw=2, label="PF, $\mu={:.2f}, \sigma={:.2f}$".format(np.mean(reso_1), np.std(reso_1)))
    plt.hist(reso_2, bins=b, histtype="step", lw=2, label="MLPF, $\mu={:.2f}, \sigma={:.2f}$".format(np.mean(reso_2), np.std(reso_2)))
    plt.yscale("log")
    plt.xlabel(r"$\eta_\mathrm{reco} - \eta_\mathrm{gen}$")
    plt.ylabel("Number of particles / bin")
    cms_label(ax)
    sample_label(sample, ax, f", {CLASS_NAMES_CMS_LATEX[pid]}")
    plt.legend(loc=(0.0, 0.7))
    plt.ylim(1, ylim)
    plt.savefig(f"{outpath}/eta_res_{CLASS_NAMES_CMS[pid]}.pdf", bbox_inches="tight")
    plt.close()


def plot_multiplicity(X, yvals, outpath, sample):
    for icls in range(1, 8):
        # Plot the particle multiplicities
        npred = np.sum(yvals["pred_cls_id"] == icls, axis=1)
        ngen = np.sum(yvals["gen_cls_id"] == icls, axis=1)
        ncand = np.sum(yvals["cand_cls_id"] == icls, axis=1)
        fig = plt.figure()
        ax = plt.axes()
        plt.scatter(ngen, ncand, marker=".", alpha=0.4, label="PF")
        plt.scatter(ngen, npred, marker=".", alpha=0.4, label="MLPF")
        a = 0.5 * min(np.min(npred), np.min(ngen))
        b = 1.5 * max(np.max(npred), np.max(ngen))
        plt.xlim(a, b)
        plt.ylim(a, b)
        plt.plot([a, b], [a, b], color="black", ls="--")
        plt.xlabel("number of truth particles")
        plt.ylabel("number of reconstructed particles")
        plt.legend(loc=4)
        cms_label(ax)
        sample_label(sample, ax, ", " + CLASS_NAMES_CMS[icls])
        plt.savefig(f"{outpath}/num_cls{icls}.pdf", bbox_inches="tight")
        plt.close()

        # Plot the sum of particle energies
        msk = yvals["gen_cls_id"][:, :, 0] == icls
        vals_gen = np.sum(np.ma.MaskedArray(yvals["gen_energy"], ~msk), axis=1)[:, 0]
        msk = yvals["pred_cls_id"][:, :, 0] == icls
        vals_pred = np.sum(np.ma.MaskedArray(yvals["pred_energy"], ~msk), axis=1)[:, 0]
        msk = yvals["cand_cls_id"][:, :, 0] == icls
        vals_cand = np.sum(np.ma.MaskedArray(yvals["cand_energy"], ~msk), axis=1)[:, 0]
        fig = plt.figure()
        ax = plt.axes()
        plt.scatter(vals_gen, vals_cand, alpha=0.2, label="PF")
        plt.scatter(vals_gen, vals_pred, alpha=0.2, label="MLPF")
        minval = min(np.min(vals_gen), np.min(vals_cand), np.min(vals_pred))
        maxval = max(np.max(vals_gen), np.max(vals_cand), np.max(vals_pred))
        plt.plot([minval, maxval], [minval, maxval], color="black")
        plt.xlim(minval, maxval)
        plt.ylim(minval, maxval)
        plt.xlabel("true $\sum E$ [GeV]")
        plt.xlabel("reconstructed $\sum E$ [GeV]")
        plt.legend(loc=4)
        cms_label(ax)
        sample_label(sample, ax, f", {CLASS_NAMES_CMS_LATEX[icls]}")
        plt.savefig(f"{outpath}/energy_cls{icls}.pdf", bbox_inches="tight")
        plt.close()


def get_distribution(yvals_f, prefix, bins, var):

    hists = []
    for pid in [13, 11, 22, 1, 2, 130, 211]:
        icls = CLASS_LABELS_CMS.index(pid)
        msk_pid = (yvals_f[prefix + "_cls_id"] == icls)
        h = bh.Histogram(bh.axis.Variable(bins))
        d = yvals_f[prefix + "_" + var][msk_pid]
        h.fill(d.flatten())
        hists.append(h)
    return hists


def plot_dist(yvals_f, var, bin, label, outpath, sample):

    hists_gen = get_distribution(yvals_f, "gen", bin, var)
    hists_cand = get_distribution(yvals_f, "cand", bin, var)
    hists_pred = get_distribution(yvals_f, "pred", bin, var)

    plt.figure()
    ax = plt.axes()
    v1 = mplhep.histplot([h[bh.rebin(2)] for h in hists_gen], stack=True, label=[class_names[k] for k in [13, 11, 22, 1, 2, 130, 211]], lw=1)
    v2 = mplhep.histplot([h[bh.rebin(2)] for h in hists_pred], stack=True, color=[x.stairs.get_edgecolor() for x in v1], lw=2, histtype="errorbar")

    legend1 = plt.legend(v1, [x.legend_artist.get_label() for x in v1], loc=(0.60, 0.44), title="true")
    legend2 = plt.legend(v2, [x.legend_artist.get_label() for x in v1], loc=(0.8, 0.44), title="pred")
    plt.gca().add_artist(legend1)
    plt.ylabel("Total number of particles / bin")
    cms_label(ax)
    sample_label(sample, ax)

    plt.yscale("log")
    plt.ylim(top=1e9)
    plt.xlabel(f"PFCandidate {label} [GeV]")
    plt.savefig(f"{outpath}/pfcand_{var}.pdf", bbox_inches="tight")
    plt.close()


def plot_eff_and_fake_rate(X_f, yvals_f, outpath, sample,
                           icls=1,
                           ivar=4,
                           ielem=1,
                           bins=np.linspace(-3, 6, 100),
                           xlabel="PFElement log[E/GeV]", log=True
                           ):

    values = X_f[:, ivar]

    hist_gen = np.histogram(values[(yvals_f["gen_cls_id"] == icls) & (X_f[:, 0] == ielem)], bins=bins)
    hist_gen_pred = np.histogram(values[(yvals_f["gen_cls_id"] == icls) & (yvals_f["pred_cls_id"] == icls) & (X_f[:, 0] == ielem)], bins=bins)
    hist_gen_cand = np.histogram(values[(yvals_f["gen_cls_id"] == icls) & (yvals_f["cand_cls_id"] == icls) & (X_f[:, 0] == ielem)], bins=bins)

    hist_pred = np.histogram(values[(yvals_f["pred_cls_id"] == icls) & (X_f[:, 0] == ielem)], bins=bins)
    hist_cand = np.histogram(values[(yvals_f["cand_cls_id"] == icls) & (X_f[:, 0] == ielem)], bins=bins)
    hist_pred_fake = np.histogram(values[(yvals_f["gen_cls_id"] != icls) & (yvals_f["pred_cls_id"] == icls) & (X_f[:, 0] == ielem)], bins=bins)
    hist_cand_fake = np.histogram(values[(yvals_f["gen_cls_id"] != icls) & (yvals_f["cand_cls_id"] == icls) & (X_f[:, 0] == ielem)], bins=bins)

    eff_mlpf = hist_gen_pred[0] / hist_gen[0]
    eff_pf = hist_gen_cand[0] / hist_gen[0]
    fake_pf = hist_cand_fake[0] / hist_cand[0]
    fake_mlpf = hist_pred_fake[0] / hist_pred[0]

    plt.figure()
    ax = plt.axes()
    mplhep.histplot(hist_gen, label="Gen", color="black")
    mplhep.histplot(hist_cand, label="PF")
    mplhep.histplot(hist_pred, label="MLPF")
    plt.ylabel("Number of PFElements / bin")
    plt.xlabel(xlabel)
    if ivar == 3:   # eta
        plt.xlim(-6, 6)
    cms_label(ax)
    sample_label(sample, ax, ", " + CLASS_NAMES_CMS[icls])
    plt.legend(loc=(0.75, 0.65))
    if log:
        plt.xscale("log")
    plt.savefig(f"{outpath}/distr_icls{icls}_ivar{ivar}.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    ax = plt.axes(sharex=ax)
    mplhep.histplot(eff_pf, bins=hist_gen[1], label="PF")
    mplhep.histplot(eff_mlpf, bins=hist_gen[1], label="MLPF")
    plt.ylim(0, 1.2)
    plt.ylabel("Efficiency")
    plt.xlabel(xlabel)
    cms_label(ax)
    sample_label(sample, ax, ", " + CLASS_NAMES_CMS[icls])
    plt.legend(loc=(0.75, 0.75))
    if log:
        plt.xscale("log")
    plt.savefig(f"{outpath}/eff_icls{icls}_ivar{ivar}.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    ax = plt.axes(sharex=ax)
    mplhep.histplot(fake_pf, bins=hist_gen[1], label="PF")
    mplhep.histplot(fake_mlpf, bins=hist_gen[1], label="MLPF")
    plt.ylim(0, 1.2)
    plt.ylabel("Fake rate")
    plt.xlabel(xlabel)
    cms_label(ax)
    sample_label(sample, ax, ", " + CLASS_NAMES_CMS[icls])
    plt.legend(loc=(0.75, 0.75))
    if log:
        plt.xscale("log")
    plt.savefig(f"{outpath}/fake_icls{icls}_ivar{ivar}.pdf", bbox_inches="tight")
    plt.close()

    #mplhep.histplot(fake, bins=hist_gen[1], label="fake rate", color="red")
#     plt.legend(frameon=False)
#     plt.ylim(0,1.4)
#     plt.xlabel(xlabel)
#     plt.ylabel("Fraction of particles / bin")


def plot_cm(yvals_f, msk_X_f, label, outpath):

    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes()

    if label == 'MLPF':
        Y = yvals_f["pred_cls_id"][msk_X_f]
    else:
        Y = yvals_f["cand_cls_id"][msk_X_f]

    cm_norm = sklearn.metrics.confusion_matrix(
        yvals_f["gen_cls_id"][msk_X_f],
        Y,
        labels=range(0, len(CLASS_LABELS_CMS)),
        normalize="true"
    )

    plt.imshow(cm_norm, cmap="Blues", origin="lower")
    plt.colorbar()

    thresh = cm_norm.max() / 1.5
    for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm_norm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm_norm[i, j] > thresh else "black", fontsize=12)

    cms_label(ax, y=1.01)
    #cms_label_sample_label(x1=0.18, x2=0.52, y=0.82)
    plt.xticks(range(len(CLASS_NAMES_CMS_LATEX)), CLASS_NAMES_CMS_LATEX, rotation=45)
    plt.yticks(range(len(CLASS_NAMES_CMS_LATEX)), CLASS_NAMES_CMS_LATEX)

    plt.xlabel(f"{label} candidate ID")
    plt.ylabel("Truth ID")
    #plt.ylim(-0.5, 6.9)
    #plt.title("MLPF trained on PF")
    plt.savefig(f"{outpath}/cm_normed_{label}.pdf", bbox_inches="tight")
    plt.close()


def distribution_icls(yvals_f, outpath):
    for icls in range(0, 8):
        fig, axs = plt.subplots(
            2, 2,
            figsize=(2 * mplhep.styles.CMS["figure.figsize"][0], 2 * mplhep.styles.CMS["figure.figsize"][1])
        )

        for ax, ivar in zip(axs.flatten(), ["pt", "energy", "eta", "phi"]):

            plt.sca(ax)

            if icls == 0:
                vals_true = yvals_f["gen_" + ivar][yvals_f["gen_cls_id"] != 0]
                vals_pf = yvals_f["cand_" + ivar][yvals_f["cand_cls_id"] != 0]
                vals_pred = yvals_f["pred_" + ivar][yvals_f["pred_cls_id"] != 0]
            else:
                vals_true = yvals_f["gen_" + ivar][yvals_f["gen_cls_id"] == icls]
                vals_pf = yvals_f["cand_" + ivar][yvals_f["cand_cls_id"] == icls]
                vals_pred = yvals_f["pred_" + ivar][yvals_f["pred_cls_id"] == icls]

            if ivar == "pt" or ivar == "energy":
                b = np.logspace(-3, 4, 61)
                log = True
            else:
                b = np.linspace(np.min(vals_true), np.max(vals_true), 41)
                log = False

            plt.hist(vals_true, bins=b, histtype="step", lw=2, label="gen", color="black")
            plt.hist(vals_pf, bins=b, histtype="step", lw=2, label="PF")
            plt.hist(vals_pred, bins=b, histtype="step", lw=2, label="MLPF")
            plt.legend(loc=(0.75, 0.75))

            ylim = ax.get_ylim()

            cls_name = CLASS_NAMES_CMS[icls] if icls > 0 else "all"
            plt.xlabel(f"{cls_name} {ivar}")

            plt.yscale("log")
            plt.ylim(10, 10 * ylim[1])

            if log:
                plt.xscale("log")
            cms_label(ax)

        plt.tight_layout()
        plt.savefig(f"{outpath}/distribution_icls{icls}.pdf", bbox_inches="tight")
        plt.close()