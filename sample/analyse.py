import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
import seaborn as sns
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
import scipy.stats as sstats

import sample.loader as loader
from redseq.loader import DatasetDCA
import redseq.utils as utils

def compute_energy_confs(x : torch.Tensor, h_parms: torch.Tensor, j_parms: torch.Tensor) -> torch.Tensor:
    M, L, q = x.shape

    # Flatten along the last two dimensions (L*q) for batch processing
    x_oh = x.reshape(M, L * q)
    bias_oh = h_parms.view(-1)  # Flatten bias
    couplings_oh = j_parms.reshape(L * q, L * q)

    # Compute energy contributions
    field = - torch.matmul(x_oh, bias_oh)  # Shape (M,)
    couplings = - 0.5 * torch.einsum('mi,ij,mj->m', x_oh, couplings_oh, x_oh)  # Shape (M,)

    return field + couplings

def get_length_distribution(batches : Dict[int, List[loader.SeqSlope]], pdf : bpdf.PdfPages, natural : str):
    fig, ax = plt.subplots(1, 1, figsize = (14,8))

    sizes = list(batches.keys())
    counts = [len(batch) for batch in batches.values()]
    nat_len = [len(str(record.seq))-str(record.seq).count('-') for record in SeqIO.parse(natural, "fasta-pearson")]

    df = pd.DataFrame(data={"size":nat_len})
    sns.histplot(data=df, x="size", kde = True, multiple="dodge", ax=ax, label="natural data")

    ax.bar(sizes, counts, label="Bins")
    ax.legend()
    ax.set_xlabel("Sequence size (nuc)")
    ax.set_ylabel("Occurence count")
    ax.set_title(f"Sequence's size distribution")

    fig.savefig(pdf, format='pdf')
    plt.close(fig)


def get_energy_diff(dataset_natural : torch.Tensor, 
                    dataset_binfile : torch.Tensor,
                    params : Dict[str, torch.Tensor], 
                    pdf : bpdf.PdfPages):
    
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))

    N, L, _ = dataset_binfile.shape

    energy_nat = compute_energy_confs(x=dataset_natural, h_parms=params["fields"], j_parms=params["couplings"]).numpy().flatten()
    energy_bin = compute_energy_confs(x=dataset_binfile, h_parms=params["fields"], j_parms=params["couplings"]).numpy().flatten()

    gaps_nat = dataset_natural[:, :, 0].sum(dim=1)
    gaps_bin = dataset_binfile[:, :, 0].sum(dim=1)

    dist_nat = gaps_nat.unique(return_counts=True)
    dist_bin = gaps_bin.unique(return_counts=True)

    all_dist = [(ngaps/L, count/count.sum()) for ngaps, count in [dist_nat, dist_bin]]

    datasets = {
        "energies_nat" : (energy_nat, gaps_nat.numpy().flatten()),
        "energies_gen" : (energy_bin, gaps_bin.numpy().flatten()),
    }

    dist_dico = dict(zip(datasets.keys(), all_dist))

    energies = []
    gaps = []
    labels = []
    for key, val in datasets.items():
        energies.extend(val[0])
        gaps.extend(val[1]/L)
        labels.extend([key] * len(val[0]))

    ngaps = []
    count = []
    label = []
    for key, val in dist_dico.items():
        ngaps.extend(val[0].numpy().flatten())
        count.extend(val[1].numpy().flatten())
        label.extend([key]* len(val[0]))

    df = pd.DataFrame({"Dataset": labels, "Energies": energies, "% of Gaps": gaps})
    df_dist = pd.DataFrame({"Dataset": label, "% of Gaps": ngaps, "Density":count})

    if N < 30000:
        sns.scatterplot(df, x = '% of Gaps', y = "Energies", hue="Dataset", ax=ax[0], alpha=0.1, legend=False)
    sns.lineplot(df, x = '% of Gaps', y = "Energies", hue="Dataset", ax=ax[0], errorbar=('ci', 95), legend=True)

    sns.lineplot(df_dist, x="% of Gaps", y="Density", hue="Dataset", ax=ax[1], markers=True, legend=True)
    
    ax[0].set_xlabel("% of gaps")
    ax[0].set_ylabel("DCA Energy")
    ax[0].legend(title="Dataset")

    fig.subplots_adjust(wspace=0.4)
    fig.savefig(pdf, format='pdf')
    plt.close(fig)
    return energy_nat, energy_bin

def show_intra_dist(intra_clust : Dict[int, List[float]], pdf : bpdf.PdfPages, k : int):
    idx = []
    dists = []
    for key, val in intra_clust.items():
        idx += [key]*len(val)
        dists += val
    
    print(len(idx), len(dists))
    dico = {"idx":idx, "dists":dists}
    df = pd.DataFrame(data=dico)

    fig, ax = plt.subplots(1, 1, figsize = (14,8))
    sns.lineplot(data=df, x="idx", y="dists", markers=True, ax=ax)
    ax.set_xlabel("Sequence size (nuc)")
    ax.set_ylabel("Jaccard distance score")
    ax.set_title(f"Jaccard distribution score depending on size with k={k}")

    fig.savefig(pdf, format='pdf')
    plt.close(fig)

def show_inter_dist(batches : Dict[int, List[loader.SeqSlope]], pdf : bpdf.PdfPages, k : int):
    mat = np.zeros((len(batches.keys()), len(batches.keys())))

    for i, (_, batch) in enumerate(batches.items()):
        
        tmp = np.array([list(seq.targets.values()) for seq in batch]).mean(axis=0)
        mat[i, i+1:] = tmp
    fig, ax = plt.subplots(1, 1, figsize = (14,8))
    sns.heatmap(data=mat, cmap="magma", cbar=True, ax=ax)
    
    ax.set_title(f"Jaccard distance matrix between sequences with k = {k}")
    fig.savefig(pdf, format='pdf')
    plt.close(fig)


def sim_nat_bin(datanat : torch.Tensor, databin : torch.Tensor, pdf : bpdf.PdfPages):
    m, s, c = datanat.shape
    N, L, q = databin.shape

    size = (torch.full((N, ), L) - databin[:, :, 0].sum(dim=1)).numpy().flatten()
    datanat = datanat.reshape(m, s*c)
    databin = databin.reshape(N, L*q)

    sim = torch.einsum("ij, kj -> ik", databin, datanat).mean(dim=1)/L

    fig, ax = plt.subplots(1, 1, figsize = (14,8))

    ax.scatter(size, sim.numpy().flatten(), marker=".")
    ax.set_xlabel("Sequence size (nuc)")
    ax.set_ylabel("Similarity score")
    ax.set_title(f"Similarity score between binfile and natural data")

    fig.savefig(pdf, format='pdf')
    plt.close(fig)
    

def main_display(path_report : str, 
                 batches : Dict[int, List[loader.SeqSlope]], 
                 intra_clust : Dict[int, List[float]], 
                 k:int,
                 natural : str,
                 outseq : str) -> bpdf.PdfPages:
    pdf = bpdf.PdfPages(path_report)

    get_length_distribution(batches=batches, pdf=pdf, natural=natural)

    show_intra_dist(intra_clust=intra_clust, pdf=pdf, k=k)
    
    params = utils.load_params(params_file=r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/sequences/scan_T_1_0/Azoarcus/non_biased/params0.json')
    natdata = DatasetDCA(path_data=natural).mat
    binfile = DatasetDCA(path_data=outseq).mat

    get_energy_diff(dataset_binfile=binfile,
                    dataset_natural=natdata,
                    params=params,
                    pdf=pdf
                    )
    
    sim_nat_bin(databin=binfile, datanat=natdata, pdf=pdf)

    #show_inter_dist(batches=batches, pdf=pdf, k=k)

    return pdf

