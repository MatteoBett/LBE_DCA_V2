#########################################################
#                        std Lib                        #
#########################################################
import os, sys, re
from typing import List
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

#########################################################
#                      Own modules                      #
#########################################################
from redseq.secondary_structures.make_struct import walk_seq
import redseq.loader as loader
import redseq.dca as dca
import redseq.utils as utils
import benchmark.benchmark_display as bdisplay
import redseq.stats as stats
import sample.analyse as analyse

sns.set_theme('paper')
matplotlib.use('pdf') 

def get_summary(all_fam_dir : str):
    template = "{0:<30} {1:<50}"
    for path, directories, files in os.walk(all_fam_dir):
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
                     indel : bool = False,
                     alphabet : str = 'rna',
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
       
    path_pdf = os.path.join(fig_dir, f'EDA_{indel}_{bin}.pdf') if bin is not None else os.path.join(fig_dir, f'EDA_{indel}.pdf')
    pdf = bpdf.PdfPages(path_pdf)

    analyse.get_energy_diff(params_path=params_path_unbiased, natfile=infile_path, binfile=chains_file_ref, pdf=pdf)
    analyse.get_energy_diff(params_path=params_path_biased, natfile=infile_path, binfile=chains_file_bias, pdf=pdf)

    """gaps_freq_heatmap(chains_file_ref=chains_file_ref, 
                      infile_path=infile_path, 
                      chains_file_bias=chains_file_bias, 
                      pdf=pdf,char=char, 
                      alphabet=alphabet)

    gaps_Fapc_heatmap(params_path_unbiased=params_path_unbiased, 
                         params_path_biased=params_path_biased,
                         pdf=pdf)

    gap_coupling_heatmap(params_path_biased=params_path_biased,
                         pdf=pdf, 
                         char=char)

    null_model = os.path.join(os.path.dirname(chains_file_bias), f"null_model0_{bin}.fasta") if bin is not None else os.path.join(os.path.dirname(chains_file_bias), f"null_model0.fasta")  
    show_contribution_profiles(
        path_null_model=null_model,
        path_natural_sequences=infile_path,
        path_chains_file_bias=chains_file_bias,
        params_path_biased=params_path_unbiased,
        pdf=pdf        
    )"""
    
    print('report finished')
    pdf.close()


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


def gaps_Fapc_heatmap(params_path_unbiased : str,
                    params_path_biased : str,
                    pdf):
    p_unbiased = utils.load_params(params_file=params_path_unbiased)
    p_biased = utils.load_params(params_file=params_path_biased)
    
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
                     chains_file_ref : str, 
                     infile_path : str,
                     chains_file_bias : str,
                     pdf,
                     char : str,
                     alphabet : str = 'rna',
                     double : bool = False):
    
    
    translate = {0: '-', 1: 'A', 2: 'U', 3: 'C', 4: 'G'}

    if double:
        infile_path = re.sub("indel", 'raw', infile_path)

    dataset_ref = loader.DatasetDCA(infile_path, alphabet=alphabet)
    dataset_biased = loader.DatasetDCA(chains_file_bias, alphabet=alphabet)
    dataset_unbiased = loader.DatasetDCA(chains_file_ref, alphabet=alphabet)
    
    fig, axes = plt.subplots(1, 1, figsize=(18,5))
    ref_count_gaps, mean_ref = dataset_ref.get_indels_info()
    biased_count_gaps, mean_bias = dataset_biased.get_indels_info()
    unbiased_count_gaps, mean_unias = dataset_unbiased.get_indels_info()

    for values in [ref_count_gaps, biased_count_gaps, unbiased_count_gaps]:
        axes.plot(list(range(len(values))), values.values())
    axes.hlines([mean_ref, mean_bias, mean_unias], xmin = 0, xmax=len(ref_count_gaps), 

                   colors=['C0', 'C1', 'C2'])
    
    axes.legend(labels=['ref', 'bias', ' unbias'])
    axes.set_title("Gaps' frequency depending on position in the MSA's sequences")
    axes.set_xlabel("Sequence position")
    axes.set_ylabel("Gap frequency")

    fig.subplots_adjust(wspace=0.2)
    fig.savefig(pdf, format='pdf')
    plt.close(fig)   
    return 
    seqref = dataset_ref.mat
    seqbiased = dataset_biased.mat
    sequnbiased = dataset_unbiased.mat

    M, L, q = seqref.shape

    f2p_ref = dca.get_freq_two_points(data=seqref).cpu()
    f2p_biased = dca.get_freq_two_points(data=seqbiased).cpu()
    f2p_unbiased = dca.get_freq_two_points(data=sequnbiased).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(f2p_unbiased[:, 0, :, 0], cmap="magma", cbar=True, ax=axes[0], vmin=0,  vmax=1)
    sns.heatmap(f2p_ref[:, 0, :, 0], cmap="magma", cbar=True, ax=axes[1], vmax=1, vmin=0)
    sns.heatmap(f2p_biased[:, 0, :, 0], cmap="magma", cbar=True, ax=axes[2], vmax=1, vmin=0)

    axes[0].set_title("Unbiased generated gaps' frequency")
    axes[1].set_title("Reference data gaps' frequency")
    axes[2].set_title("Biased generated data gaps' frequency")
    
    fig.subplots_adjust(wspace=0.2)
    fig.savefig(pdf, format='pdf')
    plt.close(fig)

    dico_f2p = dict(zip(["Unbiased", 'ref', "biased"],[f2p_unbiased, f2p_ref, f2p_biased]))
    
    for name, f2p in dico_f2p.items():
        fig, axes = plt.subplots(1, len(translate) - 1, figsize=(19, 5))
        for i in range(1, q):
            sns.heatmap(f2p[:, 0, :, i], cmap="magma", cbar=True, ax=axes[i-1])
            axes[i-1].set_title(f"Frequency heatmap {char}/{translate[i]}")
            axes[i-1].set_xlabel(f"{translate[i]} Position")
            axes[i-1].set_ylabel(f"Gaps (-) position")

            fig.suptitle(f'Frequency heatmap for {name}', fontsize=16, fontweight='bold')

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.4)
        fig.savefig(pdf, format='pdf')
        plt.close(fig)

def gap_coupling_heatmap(params_path_biased : str,
                        pdf,
                        char : str):
        
    params_biased = utils.load_params(params_file=params_path_biased, device='cpu')

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
