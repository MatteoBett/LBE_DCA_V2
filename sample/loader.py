import os, re
from dataclasses import dataclass
from typing import List, Generator, Any, Dict

from Bio import SeqIO
import RNA
import numpy as np

import redseq.secondary_structures.make_struct as struct

@dataclass
class SeqSlope:
    header : str
    seq : str
    secondary_structure : str
    encoded : List[int] | List[str]
    ss_encoded : List[int] | List[str]
    cluster_seq : list
    targets : Dict[int, List[int]]

def get_k(seq : str, ss : str):
    L1_ss = RNA.abstract_shapes(ss, 1)
    helix_count = len(re.sub("_", "", L1_ss))/2
    return int(len(seq)//helix_count)

def family_stream(family_dir : str):
    """ Yield the output of load_msa function for each family directory """
    for family_file in os.listdir(family_dir):
        yield family_file, os.path.join(family_dir, family_file, f"{family_file}.fasta")


def stream_batches(family_dir : str) -> Generator[str, Dict[int, List[SeqSlope]], Any]:
    """
    Batches the sequences contained in the family file by their size. 
    """
    for family_name, infile_path in family_stream(family_dir=family_dir):
        batches = {}
        k = []
        for index, record in enumerate(SeqIO.parse(handle=infile_path, format="fasta-pearson")):
            seq = str(record.seq)
            size = len(seq) - seq.count('-')
            if size not in batches.keys():
                seq = re.sub("-", "", seq)
                batches[size] = [SeqSlope(
                    header=record.description,
                    seq=seq,
                    secondary_structure="",
                    encoded=[],
                    ss_encoded=[],
                    cluster_seq=[],
                    targets={}
                )]
            else:
                seq = re.sub("-", "", seq)
                batches[size].append(SeqSlope(
                    header=record.description,
                    seq=seq,
                    secondary_structure="",
                    encoded=[],
                    ss_encoded=[],
                    cluster_seq=[],
                    targets={}
                ))
        yield family_name, batches