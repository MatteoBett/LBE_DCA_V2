import pandas as pd
import numpy as np

import ViennaRNA as vr
import torch
import matplotlib.pyplot as plt
from loader import DatasetDCA

ref = "GUGCCUUGCGCCGGGAAACCACGCAAGGGAUGGUGUCAAAUUCGGCGAAACCUAAGCGCCCGCCCGGGCGUAUGGCAACGCCGAGCCAAGCUUCGGCGCCUGCGCCGAUGAAGGUGUAGAGACUAGACGGCACCCACCUAAGGCAAACGCUAUGGUGAAGGCAUAGUCCAGGGAGUGGCGAAAGUCACACAAACCGG"
struct = "(((((((..((....)).)))))))...((((((....((((((...((...((((((....))))))..))...))))))(((...(.((((((....)))))).)..)))...[.[[[[[...))))))((((...(((....)))..))))......]]]]]]..((.(((((....))))).....))."
base_dir = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/sampled_seq/Azoarcus'
activity_threshold = -2.76

def viz_activity(df: pd.DataFrame):
    """
    Visualize the activity of sequences.
    """
    plt.bar(list(range(len(df["activity"]))), df["activity"], color="blue")
    plt.axhline(y=activity_threshold, color='r', linestyle='--', label='Activity Threshold')    
    plt.xlabel("Sequence Index")
    plt.ylabel("Activity")
    plt.title("Activity of Sequences")
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/infotest_fig/Azo_activity.png')
    plt.close()

def viz_entropy(entropy: list):
    """
    Visualize the Shannon entropy of sequences.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(list(range(len(entropy))), entropy, color="green")
    plt.xlabel("Sequence Index")
    plt.ylabel("Shannon Entropy")
    plt.title("Shannon Entropy of Sequences")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/infotest_fig/Azo_entropy.png')
    plt.close()

def viz_info_content_seq(entropy: list, seq: str):
    """
    Visualize the information content of a sequence.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(list(range(len(entropy))), entropy, color="purple")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Information Content (bits)")
    plt.title(f"Information Content of Sequence: {seq}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/infotest_fig/Azo_info_content.png')
    plt.close()

def shannon_entropy(p: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Shannon entropy of a probability distribution.
    """
    p = p[p > 0]  # Remove zero probabilities
    return -torch.sum(p * torch.log2(p))

def shanon_entropy_seq(file: str) -> float:
    """
    Calculate the Shannon entropy of a MSA.
    """
    dataset = DatasetDCA(file).mat[:, 3:-1, 1:]  # Exclude the first column (gaps)
    freq = dataset.sum(dim=0)/dataset.shape[0] 
    freq = freq / freq.sum(dim=0)  
    return [shannon_entropy(freq[i]).item() for i in range(freq.shape[0])], dataset
    

def load(path_tested_seq: str):
    data_all = pd.read_csv(path_tested_seq, sep=" ", header=None)
    data_all.columns = ["type", "name", "fref", "fsel", "activity"]

    return data_all

def decode_sequence(encoded_seq: torch.Tensor, alphabet: str = 'AUCG') -> str:
    """
    Decode a one-hot encoded sequence back to its string representation.
    """
    seq = ''.join([alphabet[i] for i in encoded_seq.argmax(dim=1).tolist()])
    return seq


def get_consensus(chains : torch.Tensor, path_seq) -> torch.Tensor:
    """
    Get the consensus sequence and secondary structure from a set of chains.
    """
    freq = chains.sum(dim=0)/chains.shape[0] 

    consensus = decode_sequence(freq, alphabet="AUCG")
    ss, _ = vr.fold(ref)

    return consensus, ss


def main(path_tested_seq: str = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/data/active_seq/all_tested.dat',
         path_seq : str = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/data/active_seq/all_tested.fasta'):
    """
    Main function to run the infotest.
    """  
    df = load(path_tested_seq)
    viz_activity(df=df)
    entropy, data = shanon_entropy_seq(path_seq)
    viz_entropy(entropy=entropy)
    MSA_information_content = 2*data.shape[1] - torch.tensor(entropy, dtype=torch.float32).sum().item()
    print(f"MSA Information Content: {MSA_information_content} bits")
    print(f"MSA uncertainty: {torch.tensor(entropy, dtype=torch.float32).sum().item()} bits")

    consensus, ss = get_consensus(chains=data, path_seq=path_seq)
    print(f"Consensus Sequence: {consensus}")
    print(f"Consensus Secondary Structure: {ss}")

main()