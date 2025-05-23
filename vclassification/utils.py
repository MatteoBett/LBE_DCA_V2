import torch
from torch.utils.data import DataLoader, Dataset
from random import choice, sample
from scipy.stats import binomtest
from Bio import SeqIO

NUC = {'-': 0, 'A': 1, 'U': 2, 'C': 3, 'G': 4}  # Example mapping

def pad_collate(batch, pad_value=4):
    """Pads sequences dynamically within a batch."""
    sequences, lengths = zip(*batch)
    max_length = max(lengths)
    padded_sequences = [seq + [pad_value] * (max_length - len(seq)) for seq in sequences]
    return torch.tensor(padded_sequences, dtype=torch.long)


def indices_to_one_hot(indices, num_classes):
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()


def seq_to_one_hot(sequences, pad_value=4, num_classes=5):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    return [torch.nn.functional.one_hot(torch.tensor([mapping.get(c, pad_value) for c in seq], dtype=torch.long), num_classes=num_classes).float() for seq in sequences], pad_value


def seq_to_indices(sequences, pad_value=4):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    tensor_list = [torch.tensor([mapping[c] for c in seq], dtype=torch.long) for seq in sequences]
    return tensor_list, pad_value  # Return tensor list + padding value

def indices_to_seq(tensor_batch, pad_value=4):
    """Converts a batch of tokenized sequences back to DNA strings, ignoring padding."""
    reverse_mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'U', pad_value: '-'}  # '-' represents padding
    decoded_sequences = []

    for seq in tensor_batch:
        # decoded_seq = "".join(reverse_mapping[idx.item()] for idx in seq if idx.item() != pad_value)  # Remove padding
        decoded_seq = "".join(reverse_mapping[idx.item()] for idx in seq)  # Remove padding
        decoded_sequences.append(decoded_seq)

    return decoded_sequences

def read_dca_score_dic(infile):
    "read the output of ALF-RNA"
    results = {}
    for line in open(infile):
        if not line.startswith("#"):
            posi_, posj_, nuc_i, nuc_j, nrj_ = line.strip().split()
            posi, posj = int(posi_), int(posj_)
            nrj = float(nrj_)
            if (posi, posj) not in results:
                results[(posi, posj)] = {(nuc_i, nuc_j): nrj}
            else:
                results[(posi, posj)][(nuc_i, nuc_j)] = nrj
    return results
    

def read_dca_from_text(infile):
    results = read_dca_score_dic(infile)
    positions = list(set([p for p, _ in results]))
    positions.sort()
    nb_pos = len(positions)
    h_parms = torch.zeros(size=(nb_pos, 5))
    for pi in positions:
        for ni, nii in NUC.items():
            h_parms[pi, nii] = results[pi, pi][ni, ni]
    j_parms = torch.zeros(size=(nb_pos, 5, nb_pos, 5))
    for pi in positions:
        for pj in positions[pi+1:]:
            for ni, nii in NUC.items():
                for nj, nji in NUC.items():
                    j_parms[pi, nii, pj, nji] = results[pi, pj][ni, nj]
    return h_parms, j_parms

def read_fasta(infile):
    results = {}
    for l in SeqIO.parse(infile, "fasta-pearson"):

        results[l.description] = str(l.seq)

    return results

def get_one_hot(data, num_classes=5):
    """Efficient one-hot encoding in PyTorch."""
    return torch.nn.functional.one_hot(data, num_classes=num_classes).to(dtype=torch.float32)


def msa_to_oh(msa):
    """Convert a multiple sequence alignment (MSA) to one-hot encoding."""
    if type(msa) is dict:
        msa_l = [seq_to_indices(seq)[0] for seq in msa.values()]
    elif type(msa) is list:
        msa_l = [seq_to_indices(seq)[0] for seq in msa]

    msa_tensor = torch.stack([torch.tensor(seq) for seq in msa_l]) # Shape (num_sequences, sequence_length)
    return get_one_hot(msa_tensor)  # Shape (num_sequences, sequence_length, 5)


def hamming_distance(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    return sum(nt1 != nt2 for nt1, nt2 in zip(seq1, seq2))


def random_seq(len_seq, nb_seq):
    NUC = ["A", "G", "C", "U"]
    return ["".join([choice(NUC) for _ in range(len_seq)]) for n in range(nb_seq)]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
                                                                                  