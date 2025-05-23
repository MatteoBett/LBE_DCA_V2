import os, re
from typing import List, Generator, Any
from dataclasses import dataclass

import torch
import ViennaRNA as vrna

import matplotlib.pyplot as plt
import regex
from Bio import SeqIO

from loader import DatasetDCA

@dataclass
class Family:
    """ Dataclass resulting from the parsing of Stockholm MSA """
    msa: List[str]
    family: str
    consensus_ss: str
    consensus_seq: str


def load_fam(fam_ali_file : str, outdir : str, min_size : int = 50):
    os.makedirs(outdir, exist_ok=True)
    with open(fam_ali_file, 'r', encoding="utf-8") as stock_msa:
        full_file = stock_msa.read()

    split_file = full_file.split('# STOCKHOLM 1.0')
    for piece in split_file:
        if len(piece) < 10:
            continue
        
        fam = re.findall(r'#=GF DE\s+(.*?)(?=\n)', piece)[0]
        fam = re.sub('\s|/|\.|,|\'|-|\+', '_', fam)
        msa = re.findall(r'^([A-Z]\S+)\s+([AUGC-]+)', piece, re.MULTILINE)

        if len(msa) >= 50: 
            reglen = len(max(msa, key= lambda x: len(x[1]))[1])
            
            famdir = os.path.join(outdir, fam)
            os.makedirs(famdir, exist_ok=True)
            famfile = os.path.join(famdir, f"{fam}.fasta")

            with open(famfile, 'w') as famwrite:
                for _id, seq in msa:        
                    if len(seq) == reglen:
                        famwrite.write(f">{_id}\n{seq}\n")
    return 0

#load_fam(r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/data/raw/Rfam.seed', r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/data/families')

def make_fasta(seqfile : str):
    with open(seqfile) as raw:
        raw = raw.readlines()

    with open(seqfile, 'w') as fastamode:
        for index, seq in enumerate(raw):
            fastamode.write(f">sequence_{index}\n{seq}")

#make_fasta(r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/data/raw/Azoarcus/Azoarcus.fasta')

def check_abstraction(path : str = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/data/raw/Artificial/Artificial.fasta'):
    for record in SeqIO.parse(path, "fasta-pearson"):
        ss, _ = vrna.fold(str(record.seq))
        print(vrna.abstract_shapes(ss, 5))

#generation using abstract shape level 4 or 5
#check_abstraction()

NUC = {'-':0,'A':1,'U':2,'C':3,'G':4} # Example mapping
REV_NUC = {v:k for k,v in NUC.items()}
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

def replace_index_azo(filepath : str = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/sequences/interpolation_constant/Azoarcus/biased/genseq0_0.fasta'):
    data = DatasetDCA(path_data=filepath).mat
    N, L, q = data.shape
    fi = data.sum(dim=0)
    fi_nogaps = fi[:, 1:].argmax(dim=-1) + 1
    
    gaps = fi[:, 0]/N
    n = int(L*(gaps).mean().item())

    highest = torch.topk(gaps, n).indices

    encoded_data = data.argmax(dim=-1)
    encoded_data[:, highest] = fi_nogaps[highest]

    oh = torch.nn.functional.one_hot(encoded_data, q)
    newfi = oh.sum(dim=0)/N

    save_samples(chains=oh, chains_file=filepath)               


replace_index_azo()