from typing import Dict, List

import sample.loader as loader

def encode_nucl(nucl : str):
    """ 
    Encode a nucleotide into a 2-bit integer.
    Also encode its reverse complement
    """
    encoded = (ord(nucl) >> 1) & 0b11 # Extract the two bits of the ascii code that represent the nucleotide
    rencoded = (encoded + 2) & 0b11 # Complement encoding with bit tricks. 

    return encoded, rencoded

def encode_dot_bracket(char : str):
    """ 
    Encode a dot/bracket into a 2-bit integer.
    """
    encoded = (ord(char) >> 1) & 0b11 # Extract the two bits of the ascii code that represent the nucleotide
    rencoded = (encoded + 2) & 0b11 # Complement encoding with bit tricks. 
    return encoded, rencoded

def stream_kmers(seq : str, k : int, ss_flag : bool = False):
    """
    Provide a stream of the kmers for a given sequence.
        - first loop: Add the first k-1 nucleotides to the first kmer and its reverse complement
        - yield loop: Sliding window using bit-shift to encode the entire sequence
    """
    if ss_flag:
        singleton_encoder = encode_dot_bracket
    else:
        singleton_encoder = encode_nucl

    kmer = 0
    rkmer = 0
    
    for i in range(k-1):
        nucl, rnucl = singleton_encoder(seq[i])
        kmer |= nucl << (2*(k-2-i))
        rkmer |= rnucl << (2*(i+1))

    mask = (1 << (2*(k-1))) - 1
    for i in range(k-1, len(seq)):
        nucl, rnucl = singleton_encoder(seq[i])
        kmer &= mask # Shift the kmer to make space for the new nucleotide
        kmer <<= 2 # Add the new nucleotide to the kmer
        kmer |= nucl # remove the rightmost nucleotide by side effect
        rkmer >>= 2 # Add the new nucleotide to the reverse kmer
        rkmer |= rnucl << (2*(k-1))

        yield min(kmer, rkmer) if not ss_flag else kmer


def main_kmers(seq : str, k : int, ss_flag : bool = False):
    return [kmer for kmer in stream_kmers(seq=seq, k=k, ss_flag=ss_flag)]

def encode_batches(batches : Dict[int, List[loader.SeqSlope]], k:int, ss : bool = False):
    for _, batch in batches.items():
        for elt in batch:
            elt.encoded = main_kmers(seq=elt.seq, k=k)
            if ss:
                elt.ss_encoded = main_kmers(seq=elt.secondary_structure, k=k, ss_flag=True)

                elt.ss_encoded = [x for _, x in sorted(zip(elt.encoded, elt.ss_encoded), key=lambda pair: pair[0])]
            elt.encoded.sort()

    return batches
