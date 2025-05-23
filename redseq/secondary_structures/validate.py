from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import dataclass

from Bio import SeqIO
import ViennaRNA as vrna
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf as bpdf
import numpy as np

sns.set_theme()

@dataclass
class Shape:
    ss : str
    AS1 : str
    AS2 : str
    AS3 : str
    AS4 : str
    AS5 : str

def walk_seq(seqpath : str):
    print(seqpath)
    for record in SeqIO.parse(seqpath, 'fasta'):
        tmp_struct, _ = vrna.fold(str(record.seq))
        yield tmp_struct

def load_struct(path_biased : str,
         path_unbiased : str,
         path_natural : str,
         path_artificial : str):
    
    ss_biased = [Shape(
        ss = ss,
        AS1= vrna.abstract_shapes(ss, 1),
        AS2= vrna.abstract_shapes(ss, 2),
        AS3= vrna.abstract_shapes(ss, 3),
        AS4= vrna.abstract_shapes(ss, 4),
        AS5= vrna.abstract_shapes(ss, 5)) for ss in walk_seq(path_biased)]
    ss_unbiased = [Shape(
        ss = ss,
        AS1= vrna.abstract_shapes(ss, 1),
        AS2= vrna.abstract_shapes(ss, 2),
        AS3= vrna.abstract_shapes(ss, 3),
        AS4= vrna.abstract_shapes(ss, 4),
        AS5= vrna.abstract_shapes(ss, 5)) for ss in walk_seq(path_unbiased)]
    ss_natural = [Shape(
        ss = ss,
        AS1= vrna.abstract_shapes(ss, 1),
        AS2= vrna.abstract_shapes(ss, 2),
        AS3= vrna.abstract_shapes(ss, 3),
        AS4= vrna.abstract_shapes(ss, 4),
        AS5= vrna.abstract_shapes(ss, 5)) for ss in walk_seq(path_natural)]
    ss_artificial = [Shape(
        ss = ss,
        AS1= vrna.abstract_shapes(ss, 1),
        AS2= vrna.abstract_shapes(ss, 2),
        AS3= vrna.abstract_shapes(ss, 3),
        AS4= vrna.abstract_shapes(ss, 4),
        AS5= vrna.abstract_shapes(ss, 5)) for ss in walk_seq(path_artificial)]
    
    dico = dict(zip(["ss_biased", "ss_unbiased", "ss_natural", "ss_artificial"], [ss_biased, ss_unbiased, ss_natural, ss_artificial]))

    return dico

def ssbincount(allss : Dict[str, List[Shape]], pdf : bpdf.PdfPages):
    fig, axes = plt.subplots(1, 1, figsize=(18,5))
    comparison_matrix = {}

    for key, sslist in allss.items():
        diffss1 = Counter([shape.AS1 for shape in sslist])
        diffss2 = Counter([shape.AS2 for shape in sslist])
        diffss3 = Counter([shape.AS3 for shape in sslist])
        diffss4 = Counter([shape.AS4 for shape in sslist])
        diffss5 = Counter([shape.AS5 for shape in sslist])
        
        all_diffss = [diffss1, diffss2, diffss3, diffss4, diffss5]
        axes.plot(np.arange(1, 6), [len(elt) for elt in all_diffss], label=key)
        axes.set_xlabel("Shape abstraction level")
        axes.set_ylabel("Sequences count")
        axes.set_yscale('log')
        axes.legend()

        comparison_matrix[key] = all_diffss
    
    fig.savefig(pdf, format='pdf')
    plt.close(fig)
    return comparison_matrix

def compare_shapes(dicoss : Dict[str, List[Dict[str, int]]], pdf : bpdf.PdfPages):
    ref = dicoss['ss_artificial']
    results = {}
    dicoss.pop('ss_artificial')

    for key, lst in dicoss.items():
        results[f"{key}_vs_ss_natural"] = []
        for idx, dicoshape in enumerate(lst):
            sim = 0
            tot = 0
            for shape, _ in dicoshape.items():
                if shape in ref[idx].keys():
                    sim+=1
                tot+=1
                
            results[f"{key}_vs_ss_natural"].append(sim/tot)
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 5))
    for key, val in results.items():
        ax.plot(np.arange(1, 6), val, marker='o', label=key)
        ax.set_xlabel("Shape abstraction level")
        ax.set_ylabel("Proportion of Common Shapes")
        ax.legend()
    
    fig.savefig(pdf, format='pdf')
    plt.close(fig)

def compare_shapes_number(dicoss : Dict[str, List[Dict[str, int]]], pdf : bpdf.PdfPages):
    pass
    

def main(path_biased : str,
         path_unbiased : str,
         path_natural : str,
         path_artificial : str, 
         pdf : bpdf.PdfPages):
    
    try :
        allss = load_struct(path_artificial=path_artificial,
                            path_biased=path_biased,
                            path_natural=path_natural,
                            path_unbiased=path_unbiased)   
        cdico = ssbincount(allss=allss, pdf=pdf)
        compare_shapes(cdico, pdf=pdf)

    except RuntimeError:
        return 0

    


    



