import os, re

from Bio import SeqIO   
import matplotlib.pyplot as plt

def halve_msa(infile_path : str, 
              threshold : float,
              family_file : str,
              processed_dir : str):
    
    msa = [record for record in SeqIO.parse(infile_path, 'fasta')]
    msa = sorted(msa, key= lambda x : len(x.seq) - str(x.seq).count('-'))
    longs = msa[round(len(msa)*threshold):len(msa)]
    shorts = msa[0:round(len(msa)*threshold)]

    processed_fam = os.path.join(processed_dir, family_file)
    shorts_fam = os.path.join(os.path.dirname(processed_dir), "shorts", family_file)
    os.makedirs(processed_fam, exist_ok=True)
    os.makedirs(shorts_fam, exist_ok=True)

    with open(os.path.join(processed_fam, f"{family_file}.fasta"), 'w') as processed:
        for long in longs:
            processed.write(
                f">{long.description}\n{long.seq}\n"
            )

    with open(os.path.join(shorts_fam, f"{family_file}.fasta"), 'w') as processed:
        for short in shorts:
            processed.write(
                f">{short.description}\n{short.seq}\n"
            )