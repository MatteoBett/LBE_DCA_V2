from typing import List, Dict

from Bio import SeqIO
import ViennaRNA as vrna
import torch


def walk_seq(seqpath : str):
    for record in SeqIO.parse(seqpath, 'fasta'):
        tmp_struct, _ = vrna.fold(str(record.seq))

        yield tmp_struct

def main(path_biased : str,
         path_unbiased : str,
         path_natural : str,
         path_artificial : str):
    
    ss_biased = [vrna.abstract_shapes(ss, 4) for ss in walk_seq(path_biased)]
    ss_unbiased = [vrna.abstract_shapes(ss, 4) for ss in walk_seq(path_unbiased)]
    ss_unbiased = [vrna.abstract_shapes(ss, 4) for ss in walk_seq(path_natural)]
    ss_natural = [vrna.abstract_shapes(ss, 4) for ss in walk_seq(path_artificial)]

    



