import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import seaborn as sns
import redseq.utils as utils
import pandas as pd

def mypca(dataset_nat : torch.Tensor, 
        dataset_null : torch.Tensor,
        dataset_unbiased : torch.Tensor,
        dim : int = 2):

    X_nat = dataset_nat.reshape(dataset_nat.shape[0], dataset_nat.shape[1]*dataset_nat.shape[2])
    X_null = dataset_null.reshape(dataset_null.shape[0], dataset_null.shape[1]*dataset_null.shape[2])
    X_unbiased = dataset_unbiased.reshape(dataset_unbiased.shape[0], dataset_unbiased.shape[1]*dataset_unbiased.shape[2])

    matrices = [X_nat, X_null, X_unbiased]
    X_nat_centered = X_nat - X_nat.mean(axis=0)
    _, S_nat, _ = np.linalg.svd(X_nat_centered, full_matrices=False)

    PCs = {"nat":None, "null":None, "unbiased":None}
    for index, X in enumerate(matrices):
        X_centered = X - X.mean(axis=0)
        U, _, _ = np.linalg.svd(X_centered, full_matrices=False)

        PCs[list(PCs.keys())[index]] = (U[:, :dim]) * S_nat[:dim]
        print(PCs[list(PCs.keys())[index]].shape)

    
    for index, (key, PC) in enumerate(PCs.items()):
        fig, axes = plt.subplots(1, 1, figsize=(10,10))
        if PC.shape[0] > 20000:
            heatmap, _, _ = np.histogram2d(PC[:, 0], PC[:, 1], bins=75, density=True)
            axes.imshow(heatmap.T, origin='lower', cmap='hot', aspect='auto')     
        else:
            axes.scatter(PC[:, 0], PC[:, 1], alpha=0.3)
        axes.set_xlabel("PC 1", fontsize=26)
        axes.set_ylabel("PC 2", fontsize=26)
        
        fig.savefig(f"/home/mbettiati/LBE_MatteoBettiati/report_fig/pca_{key}.png")
    return 


def get_energy_diff(dataset_natural : torch.Tensor, 
                    dataset_binfile : torch.Tensor,
                    params : Dict[str, torch.Tensor]):
    
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))

    N, L, _ = dataset_binfile.shape

    energy_nat = utils.compute_energy_confs(x=dataset_natural, h_parms=params["fields"], j_parms=params["couplings"]).numpy().flatten()
    energy_bin = utils.compute_energy_confs(x=dataset_binfile, h_parms=params["fields"], j_parms=params["couplings"]).numpy().flatten()

    gaps_nat = dataset_natural[:, :, 0].sum(dim=1)
    gaps_bin = dataset_binfile[:, :, 0].sum(dim=1)

    dist_nat = gaps_nat.unique(return_counts=True)
    dist_bin = gaps_bin.unique(return_counts=True)

    all_dist = [(ngaps/L, count/count.sum()) for ngaps, count in [dist_nat, dist_bin]]

    datasets = {
        "Energies Natural" : (energy_nat, gaps_nat.numpy().flatten()),
        "Energies DCA seq" : (energy_bin, gaps_bin.numpy().flatten()),
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
    
    ax[0].set_xlabel("% of gaps", fontsize=22)
    ax[0].set_ylabel("DCA Energy", fontsize=22)
    ax[0].legend(title="Dataset")

    ax[1].set_xlabel("% of gaps", fontsize=22)
    ax[1].set_ylabel("Density", fontsize=22)

    fig.savefig(f"/home/mbettiati/LBE_MatteoBettiati/report_fig/energy_size_dist_unbias.png", dpi=300)

    return energy_nat, energy_bin

def gaps_Fapc_heatmap(p_unbiased : str):
    
    Fapc_unbias = utils.frob_norm(p_unbiased).cpu()

    fig, axes = plt.subplots(1, 1, figsize=(15,10))
    sns.heatmap(Fapc_unbias, cmap="magma", cbar=True, ax=axes, cbar_kws={"label":"Arbitrary units", "fontsize":18})

    axes.set_xlabel("MSA position index", fontsize=22)
    axes.set_ylabel("MSA position index", fontsize=22)

    fig.savefig(f"/home/mbettiati/LBE_MatteoBettiati/report_fig/Fapc_couplings.png", dpi=300)
    return