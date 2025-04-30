import os, json
from collections import Counter
from typing import List, Dict
from typing import Tuple

import numpy as np
import torch
import torchmetrics as tm
from Bio import SeqIO
from scipy.stats import pearsonr

import redseq.dca as dca

NUC = {'-':0,'A':1,'U':2,'C':3,'G':4} # Example mapping
REV_NUC = {v:k for k,v in NUC.items()}

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


def get_freq_single_point(data, weights=None):
    if weights is not None:
        return (data * weights[:, None, None]).sum(dim=0)
    else:
        return data.mean(dim=0)


def get_freq_two_points(data, weights=None):
    M, L, q = data.shape
    data_oh = data.reshape(M, q * L)
    if weights is not None:
        we_data_oh = data_oh * weights[:, None]
    else:
        we_data_oh = data_oh * 1./M
    fij = we_data_oh.T @ data_oh  # Compute weighted sum
    return fij.reshape(L, q, L, q)


def get_one_hot(data, num_classes=5):
    """Efficient one-hot encoding in PyTorch."""
    return torch.nn.functional.one_hot(data, num_classes=num_classes).to(dtype=torch.float32)


def seq_to_indices(sequence):
    """Convert a nucleotide sequence to index tensor."""
    return torch.tensor([NUC[nt] for nt in sequence], dtype=torch.long)


def read_fasta(infile):
    """Read a FASTA file and return a dictionary of sequences."""
    results = {}
    with open(infile, 'r') as f:
        name = None
        for l in f:
            if l.startswith(">"):
                name = l.strip()[1:]
                results[name] = ""
            else:
                results[name] += l.strip()
    return results


def msa_to_oh(msa):
    """Convert a multiple sequence alignment (MSA) to one-hot encoding."""
    msa_l = [seq_to_indices(seq) for seq in msa.values()]
    msa_tensor = torch.stack(msa_l)  # Shape (num_sequences, sequence_length)
    return get_one_hot(msa_tensor)  # Shape (num_sequences, sequence_length, 5)

def family_stream(family_dir : str):
    """ Yield the output of load_msa function for each family directory """
    return [(index, family_file, os.path.join(family_dir, family_file, f"{family_file}.fasta"), len(os.listdir(family_dir))) for index, family_file in enumerate(os.listdir(family_dir))]

def get_summary(files : str, _type : str):
    template = "{0:<30} {1:<50}"
    sequences = [str(record.seq) for record in SeqIO.parse(files, 'fasta')]
    seqs = Counter("".join(sequences))
    seqs = {key : val/sum(seqs.values()) for key, val in seqs.items()}

    for key, freq in seqs.items():
        print(template.format(key, freq))
    
    if _type == 'shortest':
        obj = max(sequences, key=lambda x : x.count('-'))
        obj = obj.count("-")/len(obj)

    if _type == 'longest':
        obj = min(sequences, key=lambda x : x.count('-'))
        obj = obj.count("-")/len(obj)

    if _type == 'mean':
        obj = np.mean([seq.count('-') for seq in sequences])   
        obj = obj/len(sequences[0])

    if _type == "custom":
        obj = max(sequences, key=lambda x : x.count('-'))
        obj = obj.count("-")/len(obj)
        obj = obj*11/15

    return seqs, obj, len(sequences[0])

def save_samples(chains : torch.Tensor, chains_file : str, energies : torch.Tensor | None = None, headers : List[str] | None = None):
    N, _, _ = chains.shape
    chains=chains.argmax(dim=-1)
    seqs = ["".join([REV_NUC[i.item()] for i in encoded]) for encoded in chains]

    if headers is not None:
        with open(chains_file, 'a') as writer:
            for i in range(N):
                writer.write(f"{headers[i]}\n{seqs[i]}\n")
        return

    with open(chains_file, 'w') as writer:
        for i in range(N):
            writer.write(f">chain_{i} | DCA Energy: {energies[i].item()}\n{seqs[i]}\n")

def save_params(infile : str, params : dict[str, torch.Tensor]):
    params = {key : tensor.tolist() for key, tensor in params.items()}
    with open(infile, 'w') as json_writer:
        json.dump(params, json_writer)

def load_params(params_file : str, 
                device : str = 'cpu',
                dtype : torch.dtype = torch.float32):
    """
    Load the saved parameters of a previously trained model
    """
    with open(params_file) as trained_model:
        params = json.loads(trained_model.read())
    
    return {k : torch.tensor(v, device=device, dtype=dtype) for k,v in params.items()}


def R2_model(            
        fij: torch.Tensor,
        pij: torch.Tensor,
        fi: torch.Tensor,
        pi: torch.Tensor):
    
    L, q = fi.shape

    metric = tm.R2Score()
    fij, pij = fij.reshape(L*q, L*q), pij.reshape(L*q, L*q)
    metric.update(fij, pij)

    return metric.compute().item()


def extract_Cij_from_freq(
                        fij: torch.Tensor,
                        pij: torch.Tensor,
                        fi: torch.Tensor,
                        pi: torch.Tensor,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the lower triangular part of the covariance matrices of the data and chains starting from the frequencies.
    """
    L = fi.shape[0]
        
    # Compute the covariance matrices
    cov_data = fij - torch.einsum('ij,kl->ijkl', fi, fi)
    cov_chains = pij - torch.einsum('ij,kl->ijkl', pi, pi)

    # Extract only the entries of half the matrix and out of the diagonal blocks
    idx_row, idx_col = torch.tril_indices(L, L, offset=-1)
    fij_extract = cov_data[idx_row, :, idx_col, :].reshape(-1)
    pij_extract = cov_chains[idx_row, :, idx_col, :].reshape(-1)
    return fij_extract, pij_extract


def progressbar(iteration, total, prefix = '', suffix = '', filler = '█', printEnd = "\r") -> None:
    """ Show a progress bar indicating downloading progress """
    percent = f'{round(100 * (iteration / float(total)), 1)}'
    add = int(100 * iteration // total)
    bar = filler * add + '-' * (100 - add)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()


def two_points_correlation(fij : torch.Tensor, 
                           pij : torch.Tensor, 
                           fi : torch.Tensor, 
                           pi : torch.Tensor,
                           ):
    """
    Computes the Pearson coefficient and the slope between the two-point frequencies of data and chains.
    """
    fij_extract, pij_extract = extract_Cij_from_freq(fij, pij, fi, pi)
    stack = torch.stack([fij_extract, pij_extract])
    pearson = torch.corrcoef(stack)[0, 1].item()
        
    return pearson


def energy_corr(current_params : Dict[str, torch.Tensor],
                current_chains : torch.Tensor,
                nat_chains : torch.Tensor):
    
    
    energy_nat = dca.compute_energy_confs(x=nat_chains, h_parms=current_params["fields"], j_parms=current_params["couplings"])
    energy_chains = dca.compute_energy_confs(x=current_chains, h_parms=current_params["fields"], j_parms=current_params["couplings"])

    num_chunks = energy_nat.shape[0]
    L = energy_chains.shape[0] // num_chunks
    chunks_chains = energy_chains[:num_chunks * L].reshape(-1, L).mean(dim=1)

    stack = torch.stack([energy_nat, chunks_chains])
    pearson_energy = torch.corrcoef(stack)[0,1].item()

    return pearson_energy

def set_zerosum_gauge(coupling : torch.Tensor):
    coupling -= coupling.mean(dim=1, keepdim=True) + \
                coupling.mean(dim=3, keepdim=True) - \
                coupling.mean(dim=(1, 3), keepdim=True)
    return coupling

def frob_norm(params : dict[str, torch.Tensor], tokens : str = '-AUCG'):
    params['couplings'] = set_zerosum_gauge(params['couplings'])
    _, q = params["fields"].shape
    # Get index of the gap symbol
    gap_idx = tokens.index("-")
    
    cm_reduced = params["couplings"]
    # Take all the entries of the coupling matrix except where the gap is involved
    mask = torch.arange(q) != gap_idx
    cm_reduced = cm_reduced[:, mask, :, :][:, :, :, mask]
    
    # Compute the Frobenius norm
    print("Computing the Frobenius norm...")
    F = torch.sqrt(torch.square(cm_reduced).sum([1, 3]))
    # Set to zero the diagonal
    F = F - torch.diag(F.diag())
    # Compute the average-product corrected Frobenius norm
    Fapc = F - torch.outer(F.sum(1), F.sum(0)) / F.sum()
    
    return Fapc