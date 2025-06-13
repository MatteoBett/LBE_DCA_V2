import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import seaborn as sns
import redseq.utils as utils
import pandas as pd

import redseq.loader as loader
import redseq.utils as utils
from collections import Counter
import os

fields = torch.load(r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/fields_unbiased.pt')
couplings = torch.load(r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/couplings_unbiased.pt')
path_bin_scan = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/sampled_seq/Azoarcus/scan_T_1_0/scan_T_1_0.fasta'
path_bin_etp = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/sampled_seq/Azoarcus/fixed/Azoarcus_fixed.fasta'
path_nat = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/data/test/Azoarcus/Azoarcus.fasta'
params = {"couplings":couplings, "fields":fields}

params = utils.load_params(r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/sequences/fixed_gaps/Azoarcus/non_biased/params0.json')

databin_scan = loader.DatasetDCA(path_bin_scan).mat
databin_etp = loader.DatasetDCA(path_bin_etp).mat

datanat = loader.DatasetDCA(path_nat).mat

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
    ax[0].legend(title="Dataset", fontsize=18, title_fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=14, length=10, width=2)

    ax[1].set_xlabel("% of gaps", fontsize=22)
    ax[1].set_ylabel("Density", fontsize=22)
    ax[1].legend(title="Dataset", fontsize=18, title_fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=14, length=10, width=2)

    fig.tight_layout()
    fig.savefig(f"/home/mbettiati/LBE_MatteoBettiati/report_fig/energy_size_dist_unbias.png", dpi=300)

    return energy_nat, energy_bin

def gaps_Fapc_heatmap(p_unbiased : str):
    
    Fapc_unbias = utils.frob_norm(p_unbiased).cpu()

    fig, axes = plt.subplots(1, 1, figsize=(15,10))
    sns.heatmap(Fapc_unbias, cmap="magma", cbar=True, ax=axes, cbar_kws={"label":"Arbitrary units"})
    cbar = axes.collections[0].colorbar
    cbar.set_label("Arbitrary units", fontsize=18)

    axes.set_xlabel("MSA position index", fontsize=22)
    axes.set_ylabel("MSA position index", fontsize=22)
    axes.tick_params(axis='both', which='major', labelsize=18, length=10, width=2)
    fig.tight_layout()

    fig.savefig(f"/home/mbettiati/LBE_MatteoBettiati/report_fig/Fapc_couplings.png", dpi=300)
    return


#import redseq.loader as loader; import matplotlib.pyplot as plt; import numpy as np; import torch; import scripts.figscripts as fgs; couplings = torch.load(r'C:\Subpbiotech_cours\BT5\Stage\scripts\LBE_DCA_V2\data\raw\couplings_unbiased.pt'); fields = torch.load(r'C:\Subpbiotech_cours\BT5\Stage\scripts\LBE_DCA_V2\data\raw\fields_unbiased.pt'); path_bin = r'C:\Subpbiotech_cours\BT5\Stage\scripts\LBE_DCA_V2\output\sampled_seq\Azoarcus\scan_T_1_0\scan_T_1_0.fasta'; path_nat = r'C:\Subpbiotech_cours\BT5\Stage\scripts\LBE_DCA_V2\data\raw\Azoarcus.fasta';params = {"couplings":couplings, "fields":fields}; databin = loader.DatasetDCA(path_bin).mat; datanat = loader.DatasetDCA(path_nat).mat"""couplings = torch.load(r'C:\Subpbiotech_cours\BT5\Stage\scripts\LBE_DCA_V2\data\raw\couplings_unbiased.pt')
"""fields = torch.load(r'C:\Subpbiotech_cours\BT5\Stage\scripts\LBE_DCA_V2\data\raw\fields_unbiased.pt')
path_bin = r'C:\Subpbiotech_cours\BT5\Stage\scripts\LBE_DCA_V2\output\sampled_seq\Azoarcus\scan_T_1_0\scan_T_1_0.fasta'
path_nat = r'C:\Subpbiotech_cours\BT5\Stage\scripts\LBE_DCA_V2\data\raw\Azoarcus.fasta'
params = {"couplings":couplings, "fields":fields}
databin = loader.DatasetDCA(path_bin).mat
datanat = loader.DatasetDCA(path_nat).mat
"""

def Corrplot(dataset_nat : torch.Tensor, 
             dataset_null : torch.Tensor,
             dataset_unbiased : torch.Tensor,
             eval_method :str = "Pearson",
             ):

    nat_i, nat_ij = utils.get_freq_single_point(data=dataset_nat), utils.get_freq_two_points(data=dataset_nat)
    null_i, null_ij = utils.get_freq_single_point(data=dataset_null), utils.get_freq_two_points(data=dataset_null)
    unbiased_i, unbiased_ij = utils.get_freq_single_point(data=dataset_unbiased), utils.get_freq_two_points(data=dataset_unbiased)

    Cij_null = utils.extract_Cij_from_freq(fij=nat_ij, fi=nat_i, pi=null_i, pij=null_ij)
    Cij_unbiased = utils.extract_Cij_from_freq(fij=nat_ij, fi=nat_i, pi=unbiased_i, pij=unbiased_ij)

    nat_ij = nat_ij.ravel()

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.scatter(Cij_null[1], Cij_null[0], marker='.')

    ax.set_xlabel(r"$f_{ij}$ Null data", fontsize=16)
    ax.set_ylabel(r"$f_{ij}$ Natural data", fontsize=16)
    ax.set_title(r"Model trained at $\rho = 0.95$", fontsize=18) 
    ax.plot([Cij_null[1].min(), Cij_null[1].max()], [Cij_null[1].min(), Cij_null[1].max()], 'k--', linewidth=2, color="black")
    ax.tick_params(axis='both', which='major', labelsize=14, length=10, width=2)

    fig.tight_layout()
    fig.savefig(r'/home/mbettiati/LBE_MatteoBettiati/report_fig/null_cross_correlation.png', dpi=300)
    plt.close(fig)  

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.scatter(Cij_unbiased[1], Cij_unbiased[0], marker='.')
    ax.plot([Cij_unbiased[1].min(), Cij_unbiased[1].max()], [Cij_unbiased[1].min(), Cij_unbiased[1].max()], 'k--', linewidth=2, color="black")
    ax.tick_params(axis='both', which='major', labelsize=14, length=10, width=2)

    ax.set_xlabel(r"$f_{ij}$ Unbiased data", fontsize=16)
    ax.set_ylabel(r"$f_{ij}$ Natural data", fontsize=16)
    ax.set_title(r"Model trained at $\rho = 0.95$", fontsize=18)        

    fig.tight_layout()
    fig.savefig(r'/home/mbettiati/LBE_MatteoBettiati/report_fig/unbiased_cross_correlation.png', dpi=300)
    plt.close(fig)  


def get_length_distribution(dataset_natural : torch.Tensor,
                            dataset_binfile : torch.Tensor = databin_etp):
    fig, ax = plt.subplots(1, 1, figsize = (14,8))    
    N, L, _ = dataset_natural.shape
    n, _, _ = dataset_binfile.shape    
    len_nat = dataset_natural[:, :, 1:].sum(dim=[1,2]).flatten().tolist()
    len_bin = dataset_binfile[:, :, 1:].sum(dim=[1,2]).flatten().tolist()    
    count = Counter(len_bin)
    df = {"size":len_nat, "label":['Natural']*N}
    df = pd.DataFrame(data=df)    
    
    sns.histplot(data=df, x="size", kde = True, multiple="dodge", ax=ax, label="Natural Data")
    ax.bar(list(count.keys()), list(count.values()), label="DCA Bins")
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14, length=10, width=2)

    ax.set_xlabel("Sequence size (nuc)", fontsize=20)
    ax.set_ylabel("Occurence count", fontsize=20)    

    fig.tight_layout()
    fig.savefig(r'/home/mbettiati/LBE_MatteoBettiati/report_fig/lendist_etp.png', dpi=300)
    
    
def sim_nat_bin(datanat : torch.Tensor, databin : torch.Tensor= databin_etp):
    m, s, c = datanat.shape
    N, L, q = databin.shape    
    size = (torch.full((N, ), L) - databin[:, :, 0].sum(dim=1)).numpy().flatten()
    databin = databin.reshape(N, L*q)
    datanat = datanat.reshape(m, s*c)    
    sim = torch.einsum("ij, kj -> ik", databin, databin).mean(dim=1)/L
    sim_natbin = torch.einsum("ij, kj -> ik", databin, datanat).mean(dim=1)/L    
    fig, ax = plt.subplots(1, 1, figsize = (14,8))    
    df = {"Size":size, "Sim":sim.numpy().flatten(), "Sim nat_bin":sim_natbin.numpy().flatten()}    
    sns.lineplot(df, x = 'Size', y = "Sim", ax=ax, errorbar=('ci', 95), label="Intra-bins Identity")
    sns.lineplot(df, x = 'Size', y = "Sim nat_bin", ax=ax, errorbar=('ci', 95), label = "Natural VS DCA Identity")    
    ax.set_xlabel("Sequence size (nuc)", fontsize=20)
    ax.set_ylabel("Identity %", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14, length=10, width=2)

    ax.legend(fontsize=14)
    fig.tight_layout()
    plt.savefig(r'/home/mbettiati/LBE_MatteoBettiati/report_fig/sim_natetp.png')


def just_E(dataset_natural : torch.Tensor, 
            params : Dict[str, torch.Tensor] = params,
            dataset_binfile : torch.Tensor = databin_etp):
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    N, L, _ = dataset_binfile.shape

    energy_nat = utils.compute_energy_confs(x=dataset_natural, h_parms=params["fields"], j_parms=params["couplings"]).numpy().flatten()
    energy_bin = utils.compute_energy_confs(x=dataset_binfile, h_parms=params["fields"], j_parms=params["couplings"]).numpy().flatten()

    gaps_nat = dataset_natural[:, :, 0].sum(dim=1)
    gaps_bin = dataset_binfile[:, :, 0].sum(dim=1)

    datasets = {
        "Energies Natural" : (energy_nat, gaps_nat.numpy().flatten()),
        "Energies DCA seq" : (energy_bin, gaps_bin.numpy().flatten()),
    }

    energies = []
    gaps = []
    labels = [] 
    for key, val in datasets.items():
        energies.extend(val[0])
        gaps.extend(val[1]/L)
        labels.extend([key] * len(val[0]))

    df = pd.DataFrame({"Dataset": labels, "Energies": energies, "% of Gaps": gaps})
    if N < 30000:
        sns.scatterplot(df, x = '% of Gaps', y = "Energies", hue="Dataset", ax=ax, alpha=0.1, legend=False)
    sns.lineplot(df, x = '% of Gaps', y = "Energies", hue="Dataset", ax=ax, errorbar=('ci', 95), legend=True)

    ax.set_xlabel("% of gaps", fontsize=22)
    ax.set_ylabel("DCA Energy", fontsize=22)
    ax.legend(title="Dataset", fontsize=18, title_fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14, length=10, width=2)

    fig.tight_layout()
    fig.savefig(r"/home/mbettiati/LBE_MatteoBettiati/report_fig/energy_etp.png", dpi=300)
