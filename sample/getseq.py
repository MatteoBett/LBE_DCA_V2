import os
from typing import Tuple

from Bio import SeqIO
import torch

from redseq.loader import DatasetDCA
from redseq.utils import save_samples, progressbar


def calc_seqdiff(chains : torch.Tensor, numseq : int = 200):
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

def make_binfile(bigfasta : str, outseq_path : str, rng : Tuple[int, int]):
    start, end = rng
    chains = DatasetDCA(path_data=bigfasta, device="cpu").mat

    selection_file = os.path.join(outseq_path, f"{outseq_path.split('/')[-1]}.fasta")
    if os.path.exists(selection_file):
        os.remove(selection_file)

    for ngaps in range(start, end, 5):
        progressbar(iteration=ngaps+5, total=end)
        tmp_seq = get_seqbin(chains=chains, num_gaps=ngaps)
        indices = calc_seqdiff(chains=tmp_seq)
        headers = [f">Seq_{i.item()}_{ngaps}" for i in indices]
        selection = tmp_seq[indices]

        save_samples(chains=selection, chains_file=selection_file, headers=headers)

def make_bigfasta(genseq_path : str, tmp_output : str) -> str:
    outfile = os.path.join(tmp_output, "big_fasta.fasta")
    with open(outfile, 'w+') as bigfasta:
        for fasta in os.listdir(genseq_path):
            if fasta.startswith('genseq'):
                path_data = os.path.join(genseq_path, fasta)

                for record in SeqIO.parse(path_data, format='fasta-pearson'):
                    ngaps =str(record.seq).count('-')
                    if ngaps % 5 == 0:
                        bigfasta.write(f">{record.description} | {ngaps}\n{record.seq}\n")
    return outfile

def main_getseq(genseq_path : str, tmp_output : str, outseq_path : str):
    outfile = make_bigfasta(genseq_path=genseq_path, tmp_output=tmp_output)
    make_binfile(bigfasta=outfile, outseq_path=outseq_path, rng=(0,100))





