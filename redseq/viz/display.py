#########################################################
#                        std Lib                        #
#########################################################
import os, sys, re
from typing import List, Dict
from collections import Counter

#########################################################
#                      Dependencies                     #
#########################################################
from Bio import SeqIO
import RNA
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import scipy.stats as sstats
import matplotlib.patches as mpatches

#########################################################
#                      Own modules                      #
#########################################################
from redseq.secondary_structures.make_struct import walk_seq
import redseq.loader as loader
import redseq.utils as utils
import benchmark.benchmark_display as bdisplay
import redseq.stats as stats
import sample.analyse as analyse
import redseq.loader as loader

sns.set_theme('paper')
matplotlib.use('pdf') 

def get_summary(all_fam_dir : str):
    template = "{0:<30} {1:<50}"
    for path, _, files in os.walk(all_fam_dir):
        if files != []:
            for f in files:
                print(f.split('.')[0])
                path_file = os.path.join(path, f)
                seqs = Counter("".join([str(record.seq) for record in SeqIO.parse(path_file, 'fasta')]))
                seqs = {key : val/sum(seqs.values()) for key, val in seqs.items()}

                for key, freq in seqs.items():
                    print(template.format(key, freq))


def kde_boxplot(df : pd.DataFrame, df_col : List[str], indel : str, freq : pd.DataFrame, fig_dir : str, bin : int | None = None):
    path_pdf = os.path.join(fig_dir, f'EDA_{indel}_{bin}.pdf') if bin is not None else os.path.join(fig_dir, f'EDA_{indel}.pdf')
    pdf = bpdf.PdfPages(path_pdf)

    for col in df_col:
        if col == 'generated':
            continue
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        if col == 'ngaps' or col == "size":
            sns.histplot(df, x = col, hue = 'generated', kde = True, multiple="dodge", ax=axes[0])
            sns.boxplot(df, y = col, hue="generated", ax=axes[1])
            fig.suptitle(f'Numeric Feature : {col}', fontsize=16, fontweight='bold')
            fig.subplots_adjust(wspace=0.2)

            fig.savefig(pdf, format='pdf')
            plt.close(fig)
            continue

        if col == 'energy':
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            alignment = ['edge', 'center', 'edge']
            width_val = [-1, 1, 1]
            for ic, fcol in enumerate(freq.columns[:-1]):
                for ii in freq[fcol].index:
                    if ic < 1:
                        axes[2].bar(fcol, freq.iat[ii, ic], width=(1/len(freq[fcol].index))*width_val[ii], alpha=0.5, align=alignment[ii], color=f'C{ii}', label=f"{freq.iat[ii, -1]}")
                    else:
                        axes[2].bar(fcol, freq.iat[ii, ic], width=(1/len(freq[fcol].index))*width_val[ii], alpha=0.5, align=alignment[ii], color=f'C{ii}')
            

            sns.histplot(df, x = col, hue = 'generated', kde = True, multiple="dodge", ax=axes[0])
            #sns.scatterplot(df, x = 'ngaps', y = col, hue="generated", ax=axes[1], alpha=0.2)
            axes[2].legend(bbox_to_anchor=(1, 1), title="generated")
            sns.lineplot(df, x = 'ngaps', y = col, hue="generated", ax=axes[1], markers='o', errorbar=('ci', 100), legend=False)

            fig.suptitle(f'Numeric Feature : Sequences Energy and Nucleic acid average frequency', fontsize=16, fontweight='bold')
            fig.subplots_adjust(wspace=0.2)
            fig.savefig(pdf, format='pdf')
            plt.close(fig)
            continue

        sns.histplot(df, x = col, hue = 'generated', kde = True, multiple="dodge", ax=axes[0])
        sns.scatterplot(df, x = 'ngaps', y = col, hue="generated", ax=axes[1], alpha=0.2)
        
        fig.suptitle(f'Numeric Feature : {col}', fontsize=16, fontweight='bold')
        fig.subplots_adjust(wspace=0.2)
        
        fig.savefig(pdf, format='pdf')
        plt.close(fig)
    
    return pdf

def homology_vs_gaps(chains_file_ref : str, 
                     infile_path : str,
                     chains_file_bias : str,
                     fig_dir : str,
                     params_path_unbiased : str,
                     params_path_biased : str,
                     eval_method : str,
                     indel : bool = False,
                     constant : bool = False,
                     fixed_gaps : bool = False,
                     bin : int | None = None,
                     ):
    
    """ Computes the variation of homology depending on the number of gaps in the sequence """
    if indel:
        indel = "indel"
    elif constant:
        indel = "constant"
    elif fixed_gaps:
        indel = "fixed"
    else:
        indel = "gaps"

    char = "-"
    print("starting report")

    null_model = os.path.join(os.path.dirname(chains_file_bias), f"null_model0_{bin}.fasta") if bin is not None else os.path.join(os.path.dirname(chains_file_bias), f"null_model0.fasta")  
    path_pdf = os.path.join(fig_dir, f'EDA_{indel}_{bin}.pdf') if bin is not None else os.path.join(fig_dir, f'EDA_{indel}.pdf')
    print(path_pdf)

    pdf = bpdf.PdfPages(path_pdf)

    dataset_null= loader.DatasetDCA(null_model).mat
    dataset_biased = loader.DatasetDCA(chains_file_bias).mat
    dataset_unbiased = loader.DatasetDCA(chains_file_ref).mat
    dataset_nat = loader.DatasetDCA(infile_path).mat

    p_unbiased = utils.load_params(params_file=params_path_unbiased)
    p_biased = utils.load_params(params_file=params_path_biased)

    Corrplot(dataset_nat=dataset_nat,
             dataset_null=dataset_null,
             dataset_biased=dataset_biased,
             dataset_unbiased=dataset_unbiased,
             fig_dir=fig_dir,
             eval_method=eval_method,
             pdf=pdf)
    
    energies_nat_p_unbiased, energies_unbiased=analyse.get_energy_diff(dataset_natural=dataset_nat, dataset_binfile=dataset_unbiased, params=p_unbiased, pdf=pdf)
    energies_nat_p_biased, energies_biased=analyse.get_energy_diff(dataset_natural=dataset_nat, dataset_binfile=dataset_biased, params=p_biased, pdf=pdf)

    energy_ttest(energies_nat_p_unbiased=energies_nat_p_unbiased,
                 energies_biased=energies_biased,
                 energies_nat_p_biased=energies_nat_p_biased,
                 energies_unbiased=energies_unbiased,
                 pdf=pdf)

    mypca(dataset_nat=dataset_nat,
        dataset_null=dataset_null,
        dataset_biased=dataset_biased,
        dataset_unbiased=dataset_unbiased,
        pdf=pdf)
    
    gaps_freq_heatmap(dataset_biased=dataset_biased,
                      dataset_natural=dataset_nat,
                      dataset_unbiased=dataset_unbiased,
                      pdf=pdf)

    gaps_Fapc_heatmap(p_unbiased=p_unbiased,
                      p_biased=p_biased,
                      pdf=pdf)

    gap_coupling_heatmap(params_biased=p_biased,
                         pdf=pdf, 
                         char=char)
    
    """
    show_contribution_profiles(
        path_null_model=null_model,
        path_natural_sequences=infile_path,
        path_chains_file_bias=chains_file_bias,
        params_path_biased=params_path_unbiased,
        pdf=pdf        
    )"""
    
    print('report finished')
    pdf.close()

def mypca(dataset_nat : torch.Tensor, 
        dataset_null : torch.Tensor,
        dataset_biased : torch.Tensor,
        dataset_unbiased : torch.Tensor,
        pdf : bpdf.PdfPages,
        dim : int = 2):

    X_nat = dataset_nat.reshape(dataset_nat.shape[0], dataset_nat.shape[1]*dataset_nat.shape[2])
    X_null = dataset_null.reshape(dataset_null.shape[0], dataset_null.shape[1]*dataset_null.shape[2])
    X_biased = dataset_biased.reshape(dataset_biased.shape[0], dataset_biased.shape[1]*dataset_biased.shape[2])
    X_unbiased = dataset_unbiased.reshape(dataset_unbiased.shape[0], dataset_unbiased.shape[1]*dataset_unbiased.shape[2])

    matrices = [X_nat, X_null, X_biased, X_unbiased]
    X_nat_centered = X_nat - X_nat.mean(axis=0)
    _, S_nat, _ = np.linalg.svd(X_nat_centered, full_matrices=False)

    PCs = {"nat":None, "null":None, "biased":None, "unbiased":None}
    for index, X in enumerate(matrices):
        X_centered = X - X.mean(axis=0)
        U, _, _ = np.linalg.svd(X_centered, full_matrices=False)

        PCs[list(PCs.keys())[index]] = (U[:, :dim]) * S_nat[:dim]
        print(PCs[list(PCs.keys())[index]].shape)

    fig, axes = plt.subplots(1, 4, figsize=(16,4))
    for index, (key, PC) in enumerate(PCs.items()):
        if PC.shape[0] > 20000:
            heatmap, _, _ = np.histogram2d(PC[:, 0], PC[:, 1], bins=75, density=True)
            axes[index].imshow(heatmap.T, origin='lower', cmap='hot', aspect='auto')     
        else:
            axes[index].scatter(PC[:, 0], PC[:, 1], alpha=0.3)
        axes[index].set_title(f"PCA of {key} sequences")
        axes[index].set_xlabel("PC 1")
        axes[index].set_ylabel("PC 2")
        
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(pdf, format="pdf")
    plt.close(fig)

    return 


def energy_ttest(energies_nat_p_unbiased : np.ndarray,
                 energies_nat_p_biased : np.ndarray,
                 energies_unbiased : np.ndarray,
                 energies_biased : np.ndarray, 
                 pdf : bpdf.PdfPages):

    """ Null hypothesis tested: the two samples (natural vs generated) have identical expected values, for each bias condition. """   

    nat_unbiased = energies_nat_p_unbiased.flatten()
    gen_unbiased = energies_unbiased.flatten()
    nat_biased = energies_nat_p_biased.flatten()
    gen_biased = energies_biased.flatten()

    p_unbiased = sstats.ttest_ind(a=nat_unbiased, b=gen_unbiased, equal_var=False).pvalue
    p_biased = sstats.ttest_ind(a=nat_biased, b=gen_biased, equal_var=False).pvalue

    df = pd.DataFrame({
        "Energy": np.concatenate([gen_unbiased, gen_biased]),
        "Dataset": (["Generated"] * len(gen_unbiased) +
                    ["Generated"] * len(gen_biased)),
        "Bias": (["Unbiased"] * len(gen_unbiased) +
                 ["Biased"] * len(gen_biased))
    })
    df_nat = pd.DataFrame({
        "Energy": np.concatenate([nat_unbiased, nat_biased]),
        "Dataset": (["Natural"] * len(nat_unbiased) +
                    ["Natural"] * len(nat_biased)),
        "Bias": (["Unbiased"] * (len(nat_unbiased)) +
                 ["Biased"] * (len(nat_biased)))
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)


    for i, bias in enumerate(["Unbiased", "Biased"]):
        sns.histplot(
            data=df[df["Bias"] == bias],
            x="Energy", hue="Dataset", kde=True, bins=50, ax=axes[i],
            element="step", stat="density", palette="tab10", legend=False
        )
        sns.histplot(
            data=df_nat[df_nat["Bias"] == bias],
            x="Energy", hue="Dataset", kde=True, bins=50, ax=axes[i],
            element="step", stat="density", palette='hls', legend=False
        )
        axes[i].set_title(f"{bias} Sequences\np-value = {p_unbiased if bias=='Unbiased' else p_biased:.2e}")
        axes[i].set_xlabel("Energy")
        axes[i].set_ylabel("Density")

    # Custom legend
    legend_handles = [
        mpatches.Patch(color=sns.color_palette("tab10")[0], label="Generated"),
        mpatches.Patch(color=sns.color_palette("hls")[0], label="Natural")
    ]
    fig.legend(handles=legend_handles, loc=[0.42,0.85], ncol=2, fontsize='large')

    fig.suptitle("Independent T-Tests: Natural vs Generated Sequences", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate legend
    fig.savefig(pdf, format='pdf')
    plt.close(fig)



def Corrplot(dataset_nat : torch.Tensor, 
             dataset_null : torch.Tensor,
             dataset_biased : torch.Tensor,
             dataset_unbiased : torch.Tensor,
             fig_dir : str,
             eval_method :str,
             pdf : bpdf.PdfPages
             ):

    nat_i, nat_ij = utils.get_freq_single_point(data=dataset_nat), utils.get_freq_two_points(data=dataset_nat)
    null_i, null_ij = utils.get_freq_single_point(data=dataset_null), utils.get_freq_two_points(data=dataset_null)
    biased_i, biased_ij = utils.get_freq_single_point(data=dataset_biased), utils.get_freq_two_points(data=dataset_biased)
    unbiased_i, unbiased_ij = utils.get_freq_single_point(data=dataset_unbiased), utils.get_freq_two_points(data=dataset_unbiased)

    Cij_null = utils.extract_Cij_from_freq(fij=nat_ij, fi=nat_i, pi=null_i, pij=null_ij)
    Cij_biased = utils.extract_Cij_from_freq(fij=nat_ij, fi=nat_i, pi=biased_i, pij=biased_ij)
    Cij_unbiased = utils.extract_Cij_from_freq(fij=nat_ij, fi=nat_i, pi=unbiased_i, pij=unbiased_ij)


    corrlist = [Cij_null, Cij_biased, Cij_unbiased]
    freqlist = [null_ij.ravel(), biased_ij.ravel(), unbiased_ij.ravel()]
    nat_ij = nat_ij.ravel()

    labelist = ["null", "biased", "unbiased"]
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    fig2, axes2 = plt.subplots(1, 3, figsize=(18,5))
    for i, ax in enumerate(axes):
        ax.scatter(corrlist[i][1], corrlist[i][0],)

        ax.set_xlabel(f"Cij of {labelist[i]}")
        ax.set_ylabel("Cij Natural")
        ax.set_title(f"Model trained at {eval_method}=0.95")        

        axes2[i].scatter(freqlist[i], nat_ij, color='r')

        axes2[i].set_xlabel(f"fij of {labelist[i]}")
        axes2[i].set_ylabel("fij Natural")
        axes2[i].set_title(f"Model trained at {eval_method}=0.95")

    fig.subplots_adjust(wspace=0.4)
    fig.savefig(os.path.join(fig_dir, "Cross-correlation.png"))
    plt.close(fig)
        
    fig2.subplots_adjust(wspace=0.4)
    fig2.savefig(os.path.join(fig_dir, "Correlation_pearson.png"))
    plt.close(fig)
        

def show_contribution_profiles(path_null_model : str, 
                               path_natural_sequences : str,
                               path_chains_file_bias : str,
                               params_path_biased : str,
                               pdf : bpdf.PdfPages):
    
    print("Preparing contribution profiles")

    null_dataset = loader.DatasetDCA(path_data=path_null_model, device='cpu').mat
    natural_dataset = loader.DatasetDCA(path_data=path_natural_sequences, device='cpu').mat
    generated_chains = loader.DatasetDCA(path_data=path_chains_file_bias, device='cpu').mat

    trained_params = utils.load_params(params_file=params_path_biased, device="cpu")

    biased_cp, biased_gcp = stats.contribution_profile(seqs=generated_chains, params=trained_params)
    null_cp, null_gcp = stats.contribution_profile(seqs=null_dataset, params=trained_params)
    adata_cp, adata_gcp = stats.contribution_profile(seqs=natural_dataset, params=trained_params)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(data=adata_gcp, cmap="magma", cbar=True, ax=axes[0])
    sns.heatmap(data=null_gcp, cmap="magma", cbar=True, ax=axes[1])
    sns.heatmap(data=biased_gcp, cmap="magma", cbar=True, ax=axes[2])

    fig.suptitle("Global contribution profiles according to the trained model")
    titles = ["All Natural", "Null Model", "DCA Biased"]
    for ax, title in zip(axes, titles):
        ax.set_title(f"Contribution profile in {title} data")
        ax.set_xticks(np.arange(0.5, 5.5, dtype=np.float32))
        ax.set_xticklabels(['-', 'A', 'U', 'C', 'G'])

    fig.subplots_adjust(wspace=0.4)
    fig.savefig(pdf, format='pdf')
    plt.close(fig)

    length = []

    scores_clamped = []
    labels_clamped = []
    datasets_clamped = {
        "all seqs": bdisplay.contribution_score(samples=natural_dataset, ct_mat=adata_gcp, clamp=True),
        "null seqs": bdisplay.contribution_score(samples=null_dataset, ct_mat=null_gcp, clamp=True),
        "biased seqs": bdisplay.contribution_score(samples=generated_chains, ct_mat=biased_gcp, clamp=True)
    }

    for key, val in datasets_clamped.items():
        scores_clamped.extend(val[1].numpy().flatten())
        length.extend(val[0].numpy().flatten())
        labels_clamped.extend([key] * len(val[1].numpy().flatten()))
    
    df_clamped = pd.DataFrame({"Dataset": labels_clamped, "Contribution Score": scores_clamped, "Seq_length": length})

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.violinplot(x="Dataset", y="Contribution Score", data=df_clamped, palette="Set2", ax=ax)

    ax.set_title("Distribution des scores de contribution")
    ax.set_xlabel("Type de séquence")
    ax.set_ylabel("Score de contribution")

    plt.xticks(rotation=30)
    fig.savefig(pdf, format='pdf')
    plt.close(fig)

    grouped = df_clamped.groupby(["Seq_length", "Dataset"])['Contribution Score'].agg(["mean", "std", "count"]).reset_index()
    grouped["ci"] = 2.58 * (grouped["std"] / np.sqrt(grouped["count"]))

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    for dataset in grouped["Dataset"].unique():
        data = grouped[grouped["Dataset"] == dataset]
        ax.plot(data["Seq_length"], data["mean"], marker='o', label=f"{dataset} Mean Contribution Score")
        ax.fill_between(data["Seq_length"], data["mean"] - data["ci"], data["mean"] + data["ci"], alpha=0.2)

    ax.set_title("Mean Contribution Score by Sequence Length with Confidence Interval for Each Dataset")
    ax.set_xlabel("Sequence Length (nuc)")
    ax.set_ylabel("Contribution Score")
    ax.legend()

    plt.xticks(rotation=30)
    fig.savefig(pdf, format='pdf')
    plt.close(fig)

def mean_ci(series):
    mean = series.mean()
    ci = 2.58 * (series.std() / np.sqrt(len(series)))  # 95% confidence interval
    return mean, ci


def gaps_Fapc_heatmap(p_unbiased : str,
                      p_biased : str,
                      pdf):
    
    Fapc_unbias = utils.frob_norm(p_unbiased).cpu()
    Fapc_bias = utils.frob_norm(p_biased).cpu()

    fig, axes = plt.subplots(1, 2, figsize=(18,5))
    sns.heatmap(Fapc_unbias, cmap="magma", cbar=True, ax=axes[0])
    sns.heatmap(Fapc_bias, cmap="magma", cbar=True, ax=axes[1])

    axes[0].set_title("Unbiased generated gaps' Frobenius norm")
    axes[1].set_title("Biased generated gaps' Frobenius norm")
    fig.subplots_adjust(wspace=0.2)
    fig.savefig(pdf, format='pdf')
    plt.close(fig)   
    return

def gaps_freq_heatmap(
                     dataset_natural : torch.Tensor, 
                     dataset_biased : torch.Tensor,
                     dataset_unbiased : torch.Tensor,
                     pdf : bpdf.PdfPages):
    
    fig, axes = plt.subplots(1, 1, figsize=(18,5))

    gaps_nat = dataset_natural[:, :, 0].sum(dim=0)/dataset_natural.shape[0]
    gaps_biased = dataset_biased[:, :, 0].sum(dim=0)/dataset_biased.shape[0]
    gaps_unbiased = dataset_unbiased[:, :, 0].sum(dim=0)/dataset_unbiased.shape[0]

    names = ["dist_nat", "dist_biased", "dist_unbiased"]
        
    dist_dico = dict(zip(names, [gaps_nat, gaps_biased, gaps_unbiased]))

    dist = []
    index = []
    label = []
    for key, val in dist_dico.items():
        dist.extend(val.numpy().flatten())
        index.extend(np.arange(0, len(val), dtype=np.int32))
        label.extend([key]* len(val))

    df_dist = pd.DataFrame({"Dataset": label, "% of Gaps": dist, "position":index})

    sns.lineplot(df_dist, x="position", y="% of Gaps", hue="Dataset", ax=axes, markers=True, legend=True)

    axes.legend()
    axes.set_title("Gaps' frequency depending on position in the MSA's sequences")
    axes.set_xlabel("Sequence position")
    axes.set_ylabel("Gap frequency")

    fig.savefig(pdf, format='pdf')
    plt.close(fig)   

def gap_coupling_heatmap(params_biased : Dict[str, torch.Tensor],
                         pdf,
                         char : str):
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    Fapc_bias = utils.frob_norm(params=params_biased).cpu()

    params_biased = params_biased["couplings"]
    L, q, L, q = params_biased.shape
    gap_nuc_couplings = torch.sqrt(params_biased[:, 0, :, 1:].sum(dim=2).square()) + torch.sqrt(params_biased[:, 1:, :, 0].sum(dim=1).square())  

    sns.heatmap(params_biased[:, 0, :, 0], cmap="magma", cbar=True, ax=axes[0])        
    
    sns.heatmap(Fapc_bias, cmap="magma", cbar=True, ax=axes[1])

    sns.heatmap(gap_nuc_couplings, cmap="magma", cbar=True, ax=axes[2])  
    

    axes[0].set_title("Generated gaps' (-/-) raw couplings")
    axes[1].set_title("Generated gaps' (-/-) couplings Frobenius norm")
    axes[2].set_title("Generate gaps/nuc (-/N | N/-) couplings Frobenius norm")
    
    fig.suptitle(f'Couplings heatmap for {char}/{char}', fontsize=16, fontweight='bold')
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(pdf, format='pdf')
    plt.close(fig)
    """dico_params = dict(zip(["biased"],[params_biased]))
    
    for name, params in dico_params.items():
        fig, axes = plt.subplots(1, len(translate) - 1, figsize=(19, 5))
        mean_J = params[:, 0, :, 0:q-1].mean()
        for i in range(1, q):
            sns.heatmap(params[:, 0, :, i]-mean_J, cmap="magma", cbar=True, ax=axes[i-1])
            axes[i-1].set_title(f"Coupling heatmap {char}/{translate[i]}")
            axes[i-1].set_xlabel(f"{translate[i]} Position")
            axes[i-1].set_ylabel(f"Gaps ({char}) position")

            fig.suptitle(f'Couplings heatmap for {name}', fontsize=16, fontweight='bold')
        

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.4)
        fig.savefig(pdf, format='pdf')
        plt.close(fig)
"""
