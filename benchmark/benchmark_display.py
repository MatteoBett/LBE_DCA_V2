import os, re
from collections import Counter
from typing import Dict, List

import torch
from Bio import SeqIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
import matplotlib
import seaborn as sns
from scipy.stats import binned_statistic, ttest_ind

import redseq.viz.display as red_display
import redseq.loader as loader
import redseq.stats as stats
import redseq.utils as utils
import redseq.dca as dca

matplotlib.use('pdf') 

def basic_stats(path_null_model : str, 
                path_natural_sequences : str,
                path_chains_file_bias : str,
                fig_dir : str,
                indel : bool):
    if indel:
        indel = "indel"
    else:
        indel = "gaps"
    char = "-"
    df_dico = {'ngaps':[], "size": [], "energy" : [], 'seqtype' : []}
    path_pdf = os.path.join(fig_dir, f'Benchmark_{indel}.pdf')
    pdf = bpdf.PdfPages(path_pdf)

    freq_unbias = Counter()
    freq_bias = Counter()
    freq_ref = Counter()

    for record in SeqIO.parse(path_null_model, "fasta"):
        freq_unbias += Counter(str(record.seq))
        df_dico['ngaps'].append(str(record.seq).count(char))
        df_dico['energy'].append(round(float(record.description.split('DCA Energy: ')[1].strip()), 3))
        df_dico['size'].append(len(re.sub('-', "", str(record.seq))))
        df_dico['seqtype'].append('null_model')
        
    
    for record in SeqIO.parse(path_natural_sequences, "fasta"):
        seq = record.seq
        freq_ref += Counter(str(seq))
        df_dico['ngaps'].append(str(seq).count(char))
        df_dico['energy'].append(0)
        df_dico['size'].append(len(re.sub('-', "", str(seq))))
        df_dico['seqtype'].append('natural_seq')

    for record in SeqIO.parse(path_chains_file_bias, "fasta"):
        seq = record.seq
        freq_bias += Counter(str(seq))
        df_dico['ngaps'].append(str(seq).count(char))
        df_dico['energy'].append(round(float(record.description.split('DCA Energy: ')[1].strip()), 3))
        df_dico['size'].append(len(re.sub('-', "", str(seq))))
        df_dico['seqtype'].append('biased_dca')

    df = pd.DataFrame(data=df_dico, index=list(range(len(df_dico['ngaps']))))
    df_freq = pd.DataFrame(data=[freq_unbias,freq_ref, freq_bias]).apply(lambda x : x.apply(lambda y : y/x.sum() ), axis=1)
    df_freq["seqtype"] = ['null_model', 'natural_seq','biased_dca']

    if 'N' in df_freq:
        df_freq = df_freq.drop('N', axis=1)
    
    return pdf, df, df_freq


def kde_boxplot(df : pd.DataFrame, indel : str, freq : pd.DataFrame, fig_dir : str, pdf : bpdf.PdfPages):
    for col in df.columns:
        if col == 'seqtype':
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
            
            sns.histplot(df, x = col, hue = 'seqtype', kde = True, multiple="dodge", ax=axes[0])
            sns.scatterplot(df, x = 'ngaps', y = col, hue="seqtype", ax=axes[1], alpha=0.2)
            axes[2].legend(bbox_to_anchor=(1, 1), title="seqtype")
            sns.lineplot(df, x = 'ngaps', y = col, hue="seqtype", ax=axes[1], markers='o', errorbar=('ci', 100), legend=False)

            fig.suptitle(f'Numeric Feature : Sequences Energy and Nucleic acid average frequency', fontsize=16, fontweight='bold')
            fig.subplots_adjust(wspace=0.2)
            fig.savefig(pdf, format='pdf')
            plt.close(fig)
            continue

        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        sns.histplot(df, x = col, hue = 'seqtype', kde = True, multiple="dodge", ax=axes[0])
        sns.scatterplot(df, x = 'ngaps', y = col, hue="seqtype", ax=axes[1], alpha=0.2)
        
        fig.suptitle(f'Numeric Feature : {col}', fontsize=16, fontweight='bold')
        fig.subplots_adjust(wspace=0.2)
        
        fig.savefig(pdf, format='pdf')
        plt.close(fig)


def make_eval_fig(null_dataset : torch.Tensor,
                  long_train_genseq : torch.Tensor,
                  all_train_genseq : torch.Tensor,
                  params : Dict[str, torch.Tensor],
                  natural_seqs : torch.Tensor,
                  ax : plt.Axes):
    
    n_null, _, _ = null_dataset.shape
    n_long, _, _ = long_train_genseq.shape
    n_all, _ ,_ = all_train_genseq.shape
    n_nat, _,_ = natural_seqs.shape

    energies_null = dca.compute_energy_confs(x=null_dataset,  h_parms=params["fields"], j_parms=params["couplings"]).numpy().flatten()
    energies_long = dca.compute_energy_confs(x=long_train_genseq, h_parms=params["fields"], j_parms=params["couplings"]).numpy().flatten()
    energies_all = dca.compute_energy_confs(x=all_train_genseq, h_parms=params["fields"], j_parms=params["couplings"]).numpy().flatten()
    energies_nat = dca.compute_energy_confs(x=natural_seqs, h_parms=params["fields"], j_parms=params["couplings"]).numpy().flatten()

    gaps_null = (null_dataset[:, :, 0]).sum(dim=1)
    gaps_long = (long_train_genseq[:, :, 0]).sum(dim=1)
    gaps_all = (all_train_genseq[:, :, 0]).sum(dim=1)
    gaps_nat = (natural_seqs[:, :, 0]).sum(dim=1)

    dist_null = gaps_null.unique(return_counts=True)
    dist_long = gaps_long.unique(return_counts=True)
    dist_all = gaps_all.unique(return_counts=True)
    dist_nat = gaps_nat.unique(return_counts=True)

    all_dist = [(ngaps, count/count.sum()) for ngaps, count in [dist_null, dist_long, dist_all, dist_nat]]


    datasets = {
        "energies_null" : (energies_null, gaps_null.numpy().flatten()),
        "energies_long" : (energies_long, gaps_long.numpy().flatten()),
        "energies_all" : (energies_all, gaps_all.numpy().flatten()),
        "energies_nat" : (energies_nat, gaps_nat.numpy().flatten())
    }

    dist_dico = dict(zip(datasets.keys(), all_dist))

    energies = []
    gaps = []
    labels = []
    for key, val in datasets.items():
        energies.extend(val[0])
        gaps.extend(val[1])
        labels.extend([key] * len(val[0]))

    ngaps = []
    count = []
    label = []
    for key, val in dist_dico.items():
        ngaps.extend(val[0].numpy().flatten())
        count.extend(val[1].numpy().flatten())
        label.extend([key]*val[0].shape[0])

    df = pd.DataFrame({"Dataset": labels, "Energies": energies, "Number of Gaps": gaps})
    df_dist = pd.DataFrame({"Dataset": label, "Gap_frequency": count, "Number of Gaps": ngaps})

    sns.scatterplot(df, x = 'Number of Gaps', y = "Energies", hue="Dataset", ax=ax[0], alpha=0.1)
    sns.lineplot(df, x = 'Number of Gaps', y = "Energies", hue="Dataset", ax=ax[0], markers='o', errorbar=('ci', 95), legend=False)

    sns.lineplot(df_dist, x = 'Number of Gaps', y="Gap_frequency", markers="o", hue = 'Dataset', ax=ax[1], legend=True)
    
    ax[0].set_xlabel("Number of gaps")
    ax[0].set_ylabel("DCA Energy")
    ax[0].legend(title="Dataset")

    ax[1].set_xlabel("Number of gaps")
    ax[1].set_ylabel("Sequence count")
    ax[1].legend(title="Dataset")

    return ax

def evaluate_model(path_null_model : str, 
                   path_chains_biased_long_train : str,
                   path_chains_biased_all_train : str,
                   params_all_train : str,
                   natural_seqs_all : str,
                   pdf : bpdf.PdfPages
                   ):
    
    params_all = utils.load_params(params_file=params_all_train, device='cpu')

    null_dataset = loader.DatasetDCA(path_data=path_null_model, device='cpu').mat
    long_train_genseq = loader.DatasetDCA(path_data=path_chains_biased_long_train, device='cpu').mat
    all_train_genseq = loader.DatasetDCA(path_data=path_chains_biased_all_train, device='cpu').mat
    natural_seqs = loader.DatasetDCA(path_data=natural_seqs_all, device='cpu').mat

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes = make_eval_fig(null_dataset=null_dataset,
                  long_train_genseq=long_train_genseq,
                  all_train_genseq=all_train_genseq,
                  natural_seqs=natural_seqs,
                  params=params_all,
                  ax=axes)


    fig.suptitle(f"Mean DCA energy from the model trained on whole MSA depending on the number of gaps")

    fig.subplots_adjust(wspace=0.4)
    fig.savefig(pdf, format='pdf')
    plt.close(fig)


def show_contribution_profiles(path_null_model : str, 
                               path_natural_short : str,
                               path_chains_file_bias : str,
                               path_natural_long : str,
                               natural_seqs_all : str,
                               params_path_biased : str,
                               pdf : bpdf.PdfPages):
    
    print("Preparing contribution profiles")

    null_dataset = loader.DatasetDCA(path_data=path_null_model, device='cpu').mat
    natural_shorts_dataset = loader.DatasetDCA(path_data=path_natural_short, device='cpu').mat
    natural_longs_dataset = loader.DatasetDCA(path_data=path_natural_long, device='cpu').mat
    natural_all_dataset = loader.DatasetDCA(path_data=natural_seqs_all, device='cpu').mat
    biased_chains = loader.DatasetDCA(path_data=path_chains_file_bias, device='cpu').mat

    trained_params = utils.load_params(params_file=params_path_biased, device="cpu")

    chains_cp, chains_gcp = stats.contribution_profile(seqs=biased_chains, params=trained_params)
    null_cp, null_gcp = stats.contribution_profile(seqs=null_dataset, params=trained_params)
    sdata_cp, sdata_gcp = stats.contribution_profile(seqs=natural_shorts_dataset, params=trained_params)
    ldata_cp, ldata_gcp = stats.contribution_profile(seqs=natural_longs_dataset, params=trained_params)
    adata_cp, adata_gcp = stats.contribution_profile(seqs=natural_all_dataset, params=trained_params)

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))

    sns.heatmap(data=sdata_gcp, cmap="magma", cbar=True, ax=axes[0])
    sns.heatmap(data=ldata_gcp, cmap="magma", cbar=True, ax=axes[1])
    sns.heatmap(data=adata_gcp, cmap="magma", cbar=True, ax=axes[2])
    sns.heatmap(data=null_gcp, cmap="magma", cbar=True, ax=axes[3])
    sns.heatmap(data=chains_gcp, cmap="magma", cbar=True, ax=axes[4])

    fig.suptitle("Global contribution profiles according to the trained model")
    titles = ["Short Natural", "Long Natural", "All Natural", "Null Model", "DCA Biased"]
    for ax, title in zip(axes, titles):
        ax.set_title(f"Contribution profile in {title} data")
        ax.set_xticks(np.arange(0.5, 5.5, dtype=np.float32))
        ax.set_xticklabels(['-', 'A', 'U', 'C', 'G'])

    fig.subplots_adjust(wspace=0.4)
    fig.savefig(pdf, format='pdf')
    plt.close(fig)

    # Compute contribution scores
    scores = []
    length = []
    labels = []
    datasets = {
        "shorts seq": contribution_score(samples=natural_shorts_dataset, ct_mat=sdata_gcp),
        "long seqs": contribution_score(samples=natural_longs_dataset, ct_mat=ldata_gcp),
        "all seqs": contribution_score(samples=natural_all_dataset, ct_mat=adata_gcp),
        "null seqs": contribution_score(samples=null_dataset, ct_mat=null_gcp),
        "biased seqs": contribution_score(samples=biased_chains, ct_mat=chains_gcp)
    }

    scores_clamped = []
    labels_clamped = []
    datasets_clamped = {
        "shorts seq": contribution_score(samples=natural_shorts_dataset, ct_mat=sdata_gcp, clamp=True),
        "long seqs": contribution_score(samples=natural_longs_dataset, ct_mat=ldata_gcp, clamp=True),
        "all seqs": contribution_score(samples=natural_all_dataset, ct_mat=adata_gcp, clamp=True),
        "null seqs": contribution_score(samples=null_dataset, ct_mat=null_gcp, clamp=True),
        "biased seqs": contribution_score(samples=biased_chains, ct_mat=chains_gcp, clamp=True)
    }

    for key, val in datasets.items():
        scores.extend(val[1].numpy().flatten())
        length.extend(val[0].numpy().flatten())
        labels.extend([key] * len(val[1].numpy().flatten()))

    for key, val in datasets_clamped.items():
        scores_clamped.extend(val[1].numpy().flatten())
        labels_clamped.extend([key] * len(val[1].numpy().flatten()))
    
    df = pd.DataFrame({"Dataset": labels, "Contribution Score": scores, "Seq_length": length})

    df_clamped = pd.DataFrame({"Dataset": labels_clamped, "Contribution Score": scores_clamped, "Seq_length": length})

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    sns.violinplot(x="Dataset", y="Contribution Score", data=df, palette="Set2", ax=ax[0])
    sns.violinplot(x="Dataset", y="Contribution Score", data=df_clamped, palette="Set2", ax=ax[1])

    ax[0].set_title("Distribution des scores de contribution clamp inf -> 1.0")
    ax[0].set_xlabel("Type de séquence")
    ax[0].set_ylabel("Score de contribution")

    ax[1].set_title("Distribution des scores de contribution clamp inf -> 0.0")
    ax[1].set_xlabel("Type de séquence")
    ax[1].set_ylabel("Score de contribution")

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

    fig.savefig(pdf, format='pdf')
    plt.close(fig)

def mean_ci(series):
    mean = series.mean()
    ci = 1.96 * (series.std() / np.sqrt(len(series)))  # 95% confidence interval
    return mean, ci

def contribution_score(samples : torch.Tensor, ct_mat : torch.Tensor, clamp : bool = False):
    _, L, _ = samples.shape
    nuc_count = samples[:,:,1:].sum(dim=[1,2])
    if clamp:
        ct_mat = torch.where(ct_mat > 1.0, torch.tensor(0, dtype=torch.float32), ct_mat)
        return nuc_count, torch.einsum("nlq, lq -> n", samples, ct_mat)
    else:
        ct_mat = ct_mat.clamp(max=1.0)
        return nuc_count, torch.einsum("nlq, lq -> n", samples, ct_mat)

def compute_hamming_distance(ref : torch.Tensor, gen : torch.Tensor):
    N, l = ref.shape
    return torch.einsum("il, kl -> ikl", ref, gen).sum()/(N*l)


def msa_heatmap(path_null_model : str, 
                path_natural_sequences : str,
                path_chains_file_bias : str,
                params_path_biased : str,
                pdf : bpdf.PdfPages):
    
    null_dataset = loader.DatasetDCA(path_data=path_null_model, device='cpu').mat[:,:,0]
    natural_dataset = loader.DatasetDCA(path_data=path_natural_sequences, device='cpu').mat[:,:,0]
    biased_chains = loader.DatasetDCA(path_data=path_chains_file_bias, device='cpu').mat[:,:,0]
    """
    null_dataset = null_dataset[torch.tensor(sorted(null_dataset.sum(axis=1).tolist()), dtype=torch.int32)]
    biased_chains = biased_chains[torch.tensor(sorted(biased_chains.sum(axis=1).tolist()), dtype=torch.int32)]
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex='all')

    sns.heatmap(data=natural_dataset, cmap='magma', ax=axes[0])
    sns.heatmap(data=null_dataset, cmap='magma', ax=axes[1])
    sns.heatmap(data=biased_chains, cmap='magma', ax=axes[2])

    hamming_ref_null = compute_hamming_distance(ref=natural_dataset, gen=null_dataset)
    hamming_ref_bias = compute_hamming_distance(ref=natural_dataset, gen=biased_chains)

    fig.suptitle("Comparison of gap repartition in generated MSA respect to the null model and natural short sequences")
    axes[0].set_title("Gap repartition in natural sequences")
    axes[1].set_title(f"Gap repartition in null model with bias\nHamming distance to natural seqs: {hamming_ref_null}")
    axes[2].set_title(f"Gap repartition in biased generated sequences\nHamming distance to natural seqs: {hamming_ref_bias}")

    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def corr_heatmap(path_null_model : str, 
                path_natural_sequences : str,
                path_chains_file_bias : str,
                path_chains_biased_all_train : str,
                pdf : bpdf.PdfPages):
    
    null_dataset = loader.DatasetDCA(path_data=path_null_model, device='cpu').mat
    natural_dataset = loader.DatasetDCA(path_data=path_natural_sequences, device='cpu').mat
    biased_chains = loader.DatasetDCA(path_data=path_chains_file_bias, device='cpu').mat
    all_train = loader.DatasetDCA(path_data=path_chains_biased_all_train, device='cpu').mat
    
    fi, fij = dca.get_freq_single_point(data=natural_dataset), dca.get_freq_two_points(data=natural_dataset)

    all_freq = [{'i':dca.get_freq_single_point(data=i), 'ij':dca.get_freq_two_points(data=i)} for i in [null_dataset, biased_chains, all_train]]



def main_display(path_null_model : str, 
                path_natural_short : str,
                path_natural_long : str,
                natural_seqs_all : str,
                path_chains_file_bias : str,
                fig_dir : str,
                params_path_biased : str,
                indel : str = "gaps"):
    
    pdf, df, df_freq = basic_stats(path_chains_file_bias=path_chains_file_bias,
                                   path_null_model=path_null_model,
                                   path_natural_sequences=path_natural_short,
                                   fig_dir=fig_dir,
                                   indel=indel)
    
    print('passed basic')
    kde_boxplot(df=df, indel=indel, freq=df_freq, fig_dir=fig_dir, pdf=pdf)
    print("passed kde")

    basedir = os.path.dirname(path_chains_file_bias)
    all_train_file = os.path.join(basedir, "genseq0.fasta")
    all_train_params = os.path.join(basedir, "params0.json")

    evaluate_model(
        path_null_model=path_null_model,
        path_chains_biased_long_train=path_chains_file_bias,
        path_chains_biased_all_train=all_train_file,
        params_all_train=all_train_params,
        natural_seqs_all=natural_seqs_all,
        pdf=pdf
    )
    
    
    msa_heatmap(path_natural_sequences=path_natural_short,
                path_null_model=path_null_model,
                path_chains_file_bias=path_chains_file_bias,
                params_path_biased=params_path_biased,
                pdf=pdf)
    print("passed msa heat")

    """
    show_contribution_profiles(path_natural_short=path_natural_short,
                path_natural_long=path_natural_long,
                natural_seqs_all=natural_seqs_all,
                path_null_model=path_null_model,
                path_chains_file_bias=path_chains_file_bias,
                params_path_biased=params_path_biased,
                pdf=pdf
                )
    print("passed contribution")"""


    pdf.close()