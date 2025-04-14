import os, re
from typing import List, Generator, Any
from dataclasses import dataclass

import torch

import regex
from Bio import SeqIO


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

