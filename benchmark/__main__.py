import argparse, os, subprocess
import torch

import redseq.utils as utils
import benchmark.preprocess as preprocess
import benchmark.benchmark_display as display

def main(family_dir : str, 
         sequences_fraction : float, 
         prepdir : str, 
         fig_dir : str, 
         dca_outdir : str,
         indel : bool):
    
    if indel:
        out = 'indel'
    else:
        out = 'raw'



    for _, family_file, infile_path, _ in utils.family_stream(family_dir=family_dir):
        preprocess.halve_msa(infile_path=infile_path, threshold=sequences_fraction, family_file=family_file, processed_dir=prepdir)

    if indel:
        subprocess.run(
            ["python", '-m', "redseq", "-f", prepdir, "-o", dca_outdir, "-m", "50", "-g", '0.08', '6.5', "-s", "100", "-b", "True", "-d", "True"]
        )
    else:
        subprocess.run(
            ["python", '-m', "redseq", "-f", prepdir, "-o", dca_outdir, "-m", "50","-s", "200", "-c", 'True', '-b', 'True', '--seq_fraction', str(int(sequences_fraction*100))]
        )
    
    for _, family_file, infile_path, _ in utils.family_stream(family_dir=family_dir):
        biased_seqs = os.path.join(dca_outdir, "sequences", out, family_file.split('.')[0], "biased", f"genseq{int(sequences_fraction*100)}.fasta")
        null_model_seqs = os.path.join(dca_outdir, "sequences", out, family_file.split('.')[0], "biased", f"null_model{int(sequences_fraction*100)}.fasta")
        natural_shorts_seq = os.path.join(os.path.join(os.path.dirname(prepdir), "shorts"), family_file.split('.')[0], f"{family_file}.fasta")
        natural_longs_seq = os.path.join(os.path.join(os.path.dirname(prepdir), "longs"), family_file.split('.')[0], f"{family_file}.fasta")
        biased_params = os.path.join(dca_outdir, "sequences", out, family_file.split('.')[0], "biased", f"params{int(sequences_fraction*100)}.json")
        
        family_fig_dir = os.path.join(fig_dir, family_file.split('.')[0])
        os.makedirs(family_fig_dir, exist_ok=True)

        display.main_display(
            path_null_model=null_model_seqs,
            path_chains_file_bias=biased_seqs,
            path_natural_short=natural_shorts_seq,
            path_natural_long = natural_longs_seq,
            natural_seqs_all = infile_path,
            params_path_biased=biased_params,
            fig_dir=family_fig_dir,
            indel=indel
        )
    

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description='Samples from a DCA model.')
    
    parser.add_argument('-f', '--family_dir', help="The root directory of the data", type=str, default=r'data/test')
    parser.add_argument('-p', '--processed_dir', help="Preprocessing output directory", type=str, default=r'data/processed_input/longs')
    parser.add_argument('-o', '--output_dir', help="Redseq's output directory", type=str, default="output")
    parser.add_argument('-i', '--fig_dir', help="Directory where the report is generated", type=str, default=r'output/figures')
    parser.add_argument('-s', '--sequences_fraction', help='Fraction of the smallest sequences to take out for validation', type=float, default=0.05)
    args = parser.parse_args()
    
    family_dir = os.path.join(base_dir, args.family_dir)
    prepdir = os.path.join(base_dir, args.processed_dir)
    fig_dir = os.path.join(base_dir, args.fig_dir)
    dca_outdir = os.path.join(base_dir, args.output_dir)
    sequences_fraction = args.sequences_fraction
    indel = False

    main(family_dir= family_dir, sequences_fraction=sequences_fraction, prepdir=prepdir, dca_outdir=dca_outdir, fig_dir=fig_dir, indel=indel)

    
