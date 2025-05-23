import os, re
from typing import List, Tuple, Dict
import json

import torch
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

from redseq.loader import DatasetDCA

def get_freq_single_point(data, weights=None):
    if weights is not None:
        return (data * weights[:, None, None]).sum(dim=0)
    else:
        return data.mean(dim=0)

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

def load_params(params_file : str, 
                device : str = 'cpu',
                dtype : torch.dtype = torch.float32):
    """
    Load the saved parameters of a previously trained model
    """
    with open(params_file) as trained_model:
        params = json.loads(trained_model.read())
    
    return {k : torch.tensor(v, device=device, dtype=dtype) for k,v in params.items()}

def stream_interpolation(interpolation_dir : str, natural_seqs : str):
    biased, unbiased = os.path.join(interpolation_dir, "biased"), os.path.join(interpolation_dir, "non_biased")
    unbiased_seqs = os.path.join(unbiased, "genseq0.fasta")
    unbiased_data = DatasetDCA(path_data=unbiased_seqs).mat
    params_unbiased = os.path.join(unbiased, "params0.json")
    params = load_params(params_unbiased)
    yield "genseq0", unbiased_data, params

    natseqs = DatasetDCA(path_data=natural_seqs).mat
    yield "natural", natseqs, params

    for file in os.listdir(biased):
        if file.startswith("genseq"):
            genseq_path = os.path.join(biased, file)
            yield file.split(".")[0], DatasetDCA(path_data=genseq_path).mat, params
        
    

def energy_plot(energies : torch.Tensor, genseq : torch.Tensor, name : str, ax : plt.Axes, ax2 : plt.Axes, i : int) -> plt.Axes:
    N, L, _ = genseq.shape

    gaps = genseq[:, :, 0].sum(dim=1)
    datasets = {
        name : (energies, gaps.numpy().flatten()),
    }

    gfreq = get_freq_single_point(data=genseq)[:,0].numpy().flatten()

    energies = []
    gaps = []
    labels = []
    for key, val in datasets.items():
        energies.extend(val[0].numpy().flatten())
        gaps.extend(val[1]/L)
        labels.extend([key] * len(val[0]))

    df = pd.DataFrame({"Dataset": labels, "Energies": energies, "% of Gaps": gaps})
    sns.lineplot(df, x = '% of Gaps', y = "Energies", ax=ax, errorbar=('ci', 95), color=f"C{i}", label=name)
    ax.legend()

    ax2.plot(gfreq, color=f"C{i}", label=name)
    ax2.legend(loc=(1.03, 0.5))
    return ax


def main_eda(interpolation_dir : str, natseqs : str, fig_dir : str):
    path_pdf = os.path.join(fig_dir, f'EDA_interpolation.pdf')
    pdf = bpdf.PdfPages(path_pdf)
    fig, ax = plt.subplots(1, 1, figsize=(18, 5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(18, 5))

    for i, (name_file, genseq, params) in enumerate(stream_interpolation(interpolation_dir=interpolation_dir, natural_seqs=natseqs)):
        E_dca = compute_energy_confs(x=genseq, h_parms=params["fields"], j_parms=params["couplings"])
        ax = energy_plot(energies=E_dca, genseq=genseq, name=name_file, ax=ax, ax2=ax2, i=i)

    fig.savefig(pdf, format='pdf')
    plt.close(fig)
    fig2.savefig(pdf, format="pdf")
    plt.close(fig2)


    pdf.close()


