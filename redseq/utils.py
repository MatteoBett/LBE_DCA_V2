import os, json
from collections import Counter
from typing import List
from typing import Tuple

import numpy as np
import torch
from Bio import SeqIO

NUC = {'-':0,'A':1,'U':2,'C':3,'G':4} # Example mapping
REV_NUC = {v:k for k,v in NUC.items()}


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

def get_summary(files : str):
    template = "{0:<30} {1:<50}"
    sequences = [str(record.seq) for record in SeqIO.parse(files, 'fasta')]
    seqs = Counter("".join(sequences))
    seqs = {key : val/sum(seqs.values()) for key, val in seqs.items()}

    for key, freq in seqs.items():
        print(template.format(key, freq))
    
    shortest = max(sequences, key=lambda x : x.count('-'))
    return seqs, shortest.count("-")/len(shortest), len(sequences[0])

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
                device : str = 'cuda:0',
                dtype : torch.dtype = torch.float32):
    """
    Load the saved parameters of a previously trained model
    """
    with open(params_file) as trained_model:
        params = json.loads(trained_model.read())
    
    return {k : torch.tensor(v, device=device, dtype=dtype) for k,v in params.items()}


def extract_Cij_from_freq(
                        fij: torch.Tensor,
                        pij: torch.Tensor,
                        fi: torch.Tensor,
                        pi: torch.Tensor,
                        ) -> Tuple[float, float]:
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