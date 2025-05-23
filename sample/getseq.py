import os
from typing import Tuple

from Bio import SeqIO
import torch
import numpy as np
import matplotlib.pyplot as plt

from redseq.loader import DatasetDCA
from redseq.utils import save_samples, progressbar


def calc_seqdiff(chains : torch.Tensor, numseq : int = 500):
    n, L, q = chains.shape
    chains = chains.argmax(dim=-1)

    def get_sequence_weight(s: torch.Tensor, data: torch.Tensor, L: int):
        seq_id = torch.sum(s == data, dim=1) / L
        return seq_id
    
    distmat = torch.vstack([get_sequence_weight(s, chains, L) for s in chains]).mean(dim=1)
    return torch.topk(distmat, k=numseq, largest=False).indices

def get_seqbin(chains : torch.Tensor, num_gaps : int) -> torch.Tensor:
    gapdist = chains[:, :, 0].sum(dim=1)
    mask = torch.where(gapdist == num_gaps, 1, 0).nonzero()
    return chains[mask].squeeze(dim=1) 

def make_binfile(bigfasta : str, bignull : str, outseq_path : str, rng : Tuple[int, int]):
    start, end = rng
    chains = DatasetDCA(path_data=bigfasta, device="cpu").mat
    chains_null = DatasetDCA(path_data=bignull, device="cpu").mat


    selection_file = os.path.join(outseq_path, f"{outseq_path.split('/')[-1]}.fasta")
    selection_null = os.path.join(outseq_path, f"{outseq_path.split('/')[-1]}_null.fasta")

    if os.path.exists(selection_file):
        os.remove(selection_file)
    if os.path.exists(selection_null):
        os.remove(selection_null)

    for ngaps in range(start, end, 5):
        progressbar(iteration=ngaps+5, total=end)
        tmp_seq = get_seqbin(chains=chains, num_gaps=ngaps)
        tmp_null = get_seqbin(chains=chains_null, num_gaps=ngaps)
        try:
            """indices = calc_seqdiff(chains=tmp_seq)
            indices_null = calc_seqdiff(chains=tmp_null)"""

            indices = np.random.choice(list(range(len(tmp_seq))), size=500, replace=False)
            indices_null = np.random.choice(list(range(len(tmp_null))), size=500, replace=False)

        except ValueError:
            continue

        headers = [f">Seq_{i.item()}_{ngaps}" for i in indices]
        selection = tmp_seq[indices]
        save_samples(chains=selection, chains_file=selection_file, headers=headers)

        headers = [f">Seq_{i.item()}_{ngaps}" for i in indices_null]
        selection = tmp_null[indices_null]
        save_samples(chains=selection, chains_file=selection_null, headers=headers)
    
    return selection_file, selection_null

def make_bigfasta(genseq_path : str, tmp_output : str) -> str:
    outfile = os.path.join(tmp_output, "big_fasta.fasta")
    outnull = os.path.join(tmp_output, "big_null.fasta")

    with open(outfile, 'w+') as bigfasta:
        for fasta in os.listdir(genseq_path):
            if fasta.startswith('genseq'):
                path_data = os.path.join(genseq_path, fasta)

                for record in SeqIO.parse(path_data, format='fasta-pearson'):
                    ngaps =str(record.seq).count('-')
                    if ngaps % 5 == 0:
                        bigfasta.write(f">{record.description} | {ngaps}\n{record.seq}\n")
                
    with open(outnull, 'w+') as bignull:
        for nullname in os.listdir(genseq_path):
            if nullname.startswith('null_model'):
                path_data = os.path.join(genseq_path, nullname)
                for record in SeqIO.parse(path_data, format='fasta-pearson'):
                    ngaps =str(record.seq).count('-')
                    if ngaps % 5 == 0:
                        bignull.write(f">{record.description} | {ngaps}\n{record.seq}\n")

    return outfile, outnull

def main_getseq(genseq_path : str, tmp_output : str, outseq_path : str, rng : Tuple[int, int]):
    outfile, outnull = make_bigfasta(genseq_path=genseq_path, tmp_output=tmp_output)
    return make_binfile(bigfasta=outfile, bignull=outnull, outseq_path=outseq_path, rng=rng)





