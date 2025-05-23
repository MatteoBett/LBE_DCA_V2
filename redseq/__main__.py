"""Code derived from XXX
"""

import argparse, os
from typing import List, Tuple
from multiprocessing import Pool

import torch

import redseq.dca as dca
import redseq.utils as utils
from redseq.loader import DatasetDCA
import benchmark.benchmark_display as bm_display

def premain(famtuple : Tuple[str], do_one : bool = False):
        torch.set_num_threads(1)
        index, family_file, infile_path, tot = famtuple
        
        gap_fraction = gaps_fraction[index]
        seqs = utils.get_summary(infile_path)
        gap_fraction = seqs['-'] + gap_fraction*seqs['-'] 
        
        if bias:
            family_outdir = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased")
            print("Gap fraction targetted", gap_fraction)

        else:
            family_outdir = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased")
        
        os.makedirs(family_outdir, exist_ok=True)
        if run_generation:
            main(
                infile=infile_path,
                max_steps=max_steps,
                min_pearson=target_pearson,
                sample_it=sample_it,
                lr=lr,
                family_outdir=family_outdir,
                gap_fraction=gap_fraction,
                bias_flag=bias,
                indel=indel
            )

        biased_seqs = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased", f"genseq{seqs_fraction}.fasta")
        unbiased_seqs = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased",f"genseq{seqs_fraction}.fasta")
        biased_params = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased", f"params{seqs_fraction}.json")
        unbiased_params = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased", f"params{seqs_fraction}.json")
        null_model_seqs = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased", f"null_model{seqs_fraction}.fasta")
        natural_shorts_seq = os.path.join(os.path.join(os.path.dirname(family_dir), "shorts"), family_file.split('.')[0], f"{family_file}.fasta")


        family_fig_dir = os.path.join(fig_dir, family_file.split('.')[0])
        os.makedirs(family_fig_dir, exist_ok=True)

        bm_display.main_display(
        path_null_model=null_model_seqs,
        path_chains_file_bias=biased_seqs,
        path_natural_sequences=natural_shorts_seq,
        params_path_biased=biased_params,
        fig_dir=family_fig_dir,
        indel=indel
    )
    
        if do_one:
            return


def main(infile : str, 
         max_steps : int, 
         min_eval : float, 
         lr : float, 
         sample_it : int, 
         family_outdir_biased : str, 
         family_outdir_unbiased : str, 
         gap_fraction : float, 
         bias_flag : bool, 
         indel : bool,
         seq_fraction : int,
         constant_bias : bool,
         fixed_gaps : bool,
         eval_method : str,
         adaptative : bool,
         plotting : bool,
         family_fig_dir : str,
         nseqs : int,
         scan : bool,
         scanrange : List[int],
         beta : float,
         interpolation_targets : List[float] | None = None,
         full_interpolation : bool | None = None,
         ):
    
    params_dca_biased = os.path.join(family_outdir_biased, f"params{seq_fraction}.json") 
    sample_seq_biased = os.path.join(family_outdir_biased, f"genseq{seq_fraction}.fasta") 
    params_dca_unbiased =  os.path.join(family_outdir_unbiased, f"params{seq_fraction}.json") 
    sample_seq_unbiased = os.path.join(family_outdir_unbiased, f"genseq{seq_fraction}.fasta") 
    dataset = DatasetDCA(path_data=infile, 
                         params_file_unbiased=params_dca_unbiased,
                         params_file_biased=params_dca_biased,
                         chains_biased_file=sample_seq_biased,
                         chains_unbiased_file=sample_seq_unbiased,
                         device='cpu')

    n_chains = nseqs
    nb_sweep = 10
    beta = beta
    dca.fit_model(dataset=dataset, 
                  max_step=max_steps, 
                  min_eval=min_eval,
                  N=n_chains,
                  nb_gibbs=nb_sweep,
                  lr=lr, 
                  eval_method=eval_method)
    
    dca.sample_trained(dataset=dataset,
                       num_gen=n_chains, 
                       max_sweeps=nb_sweep,
                       sample_it=sample_it,
                       min_eval=min_eval,
                       gap_fraction=gap_fraction,
                       bias_flag=bias_flag,
                       indel=indel,
                       beta=beta,
                       seq_fraction=seq_fraction,
                       constant_bias=constant_bias,
                       fixed_gaps=fixed_gaps,
                       eval_method=eval_method,
                       adaptative=adaptative,
                       interpolation_targets=interpolation_targets,
                       full_interpolation=full_interpolation,
                       plotting=plotting,
                       family_fig_dir=family_fig_dir,
                       scan=scan,
                       scanrange=scanrange)

def save_params():
    params = utils.load_params(params_file=r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/sequences/interpolation_constant/Azoarcus/non_biased/params0.json') 

    torch.save(params["couplings"], f=r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/couplings_unbiased.pt')
    torch.save(params["fields"], f=r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/fields_unbiased.pt')

if __name__ == "__main__":
    save_params()
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description='Samples from a DCA model.')

    parser.add_argument('-f', '--family_dir', help="The root directory of the data", type=str, default=r'data/test')
    parser.add_argument('-o', '--output', help="Output directory", type=str, default='output')
    parser.add_argument('-a', '--alphabet', help="Alphabet of tokens used in sequences (e.g. -AUCG or -ATCG)", type=str, default='-AUCG')
    parser.add_argument('-t', '--target_pearson', help='Targetted pearson correlation with the input sequences', type=float, default=0.95)
    parser.add_argument('-m', "--max_steps", help='Maximum number of iterations to make converge the model. Halts the model if the target pearson is not reached' , type=int, default=250)
    parser.add_argument('-lr', "--learning_rate", help='Learning rate of the DCA model', type=float, default=0.015)
    parser.add_argument('-g', '--gaps_fraction', help='Targetted average gap fraction in the generated sequences', type=float, nargs='+', default=[])
    parser.add_argument('-s', '--sample_iterations', help="The maximum number of sample iterations to perform", type=int, default=150)
    parser.add_argument('-e', '--eval_method', help='Choose evaluation method for convergence. Currently supported: R2 and pearson. Default: pearson', type=str, default='pearson')
    parser.add_argument('-n', '--nseqs', help='Number of sequences to generate', type=int, default=10000)

    parser.add_argument('--do_one', help="If True, only one family will be processed", type=bool, default=False)
    parser.add_argument('-p', '--plotting', help='If True then the pdf report is generated', type=bool, default=True)
    parser.add_argument('-r', '--run_generation', help='If True then sequence generation is run', type=bool, default=True)
    parser.add_argument("-b", '--bias', help='Bias the gaps distribution', type=bool, default=False)
    parser.add_argument("-c", '--constant_bias', help="Apply a constant bias on the data", type=bool, default=False)
    parser.add_argument("-d", '--indel', help="Indicate whether to discriminate between types of gaps.", type=bool, default=False)
    parser.add_argument("--temperature", help="Choose temperature value.", type=float, default=1.0)
    parser.add_argument("--scan", help="Indicates scan mode of bias", type=bool, default=False)
    parser.add_argument("--scanrange", help="Linear scan in the given range of values followed by the number of desired values.", type=float, nargs=3, default=[-7, 2, 50])
    parser.add_argument('--adaptative', help="Adaptative gap bias", type=bool, default=False)
    parser.add_argument('--seq_fraction', help="Indicates the fraction of sequences used in the modelling", type=int, default=0)
    parser.add_argument('--fixed_gaps', help="Fixes the most occurring gaps in the alignment corresponding to the designated bias and explore the sequence space for those.", type=bool, default=False)
    parser.add_argument('--full_interpolation', help="Passes the program in interpolation mode and repeats generation until all bins are passed", type=bool, default=False)
    parser.add_argument("--interpolation_bin_size", help="Indicates in absolute values the bins of gaps to consider", type=int, default=5)
    parser.add_argument("--interpolation_range", help="Indicates the targetted % range of gaps bias for interpolation", type=int, nargs=2, default=[0,50])
    args = parser.parse_args()

    family_dir = os.path.join(base_dir, args.family_dir)
    outdir = os.path.join(base_dir, args.output)
    fig_dir = os.path.join(outdir, "figures")

    alphabet = args.alphabet
    target_pearson = args.target_pearson
    max_steps = args.max_steps
    nseqs = args.nseqs
    lr = args.learning_rate
    gaps_fraction = args.gaps_fraction 
    seqs_fraction = args.seq_fraction
    sample_it = args.sample_iterations
    constant_bias = args.constant_bias

    do_one = args.do_one
    plotting = args.plotting
    run_generation = args.run_generation
    full_interpolation = args.full_interpolation
    interpolation_bins = args.interpolation_bin_size
    fixed_gaps = args.fixed_gaps  

    bias = args.bias 
    indel = args.indel
    adaptative = args.adaptative
    famlist = utils.family_stream(family_dir=family_dir)
    use_min = False
    eval_method = args.eval_method
    scan = args.scan
    scanrange = args.scanrange
    interpolation_start, interpolation_end = args.interpolation_range
    interpolation_targets = None
    
    beta = args.temperature

    if bias:
        if len(gaps_fraction) <= 1:
            use_min = True
    else:
        gaps_fraction = [0]*len(famlist)

    if (constant_bias and not full_interpolation):
        out = "constant"
    elif (fixed_gaps and not full_interpolation):
        out = "fixed_gaps"
    elif (indel and not full_interpolation):
        out = 'indel'
    elif adaptative and not full_interpolation:
        out = "adaptative"
    elif scan and not full_interpolation:
        out = f"scan_T_{"_".join(str(beta).split("."))}"
    elif indel and full_interpolation:
        out = 'indel'
    elif full_interpolation and adaptative:
        out = "interpolation_adaptative"
    elif full_interpolation and fixed_gaps:
        out = "interpolation_fixed"
    elif full_interpolation and constant_bias:
        out = "interpolation_constant"
    else:
        out = 'raw'

    print(out)
    parallel = False
    
    if parallel:
        with Pool(2) as p:
            print(p.map(premain, famlist))

    for index, family_file, infile_path, tot in utils.family_stream(family_dir=family_dir):
        seqs, shortest, sizeseq = utils.get_summary(infile_path, 'custom')
        if use_min:
            gap_fraction = shortest
        else:
            gap_fraction = gaps_fraction[index]
            gap_fraction = seqs['-'] + gap_fraction*seqs['-'] 

        if full_interpolation:
            interpolation_targets = [val/sizeseq for val in range(interpolation_start, interpolation_end, interpolation_bins)]
        
        family_outdir_biased = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased")
        print("Gap fraction targetted", gap_fraction)
        family_outdir_unbiased = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased")
        if plotting:
            family_fig_dir = os.path.join(fig_dir, family_file.split('.')[0], out)
            os.makedirs(family_fig_dir, exist_ok=True)

        os.makedirs(family_outdir_biased, exist_ok=True)
        os.makedirs(family_outdir_unbiased, exist_ok=True)

        if run_generation:
            main(
                infile=infile_path,
                max_steps=max_steps,
                min_eval=target_pearson,
                sample_it=sample_it,
                lr=lr,
                family_outdir_biased=family_outdir_biased,
                family_outdir_unbiased=family_outdir_unbiased,
                gap_fraction=gap_fraction,
                bias_flag=bias,
                indel=indel,
                seq_fraction=seqs_fraction,
                constant_bias=constant_bias,
                fixed_gaps=fixed_gaps,
                eval_method=eval_method,
                adaptative=adaptative,
                full_interpolation=full_interpolation,
                interpolation_targets=interpolation_targets,
                plotting=plotting,
                family_fig_dir=family_fig_dir,
                nseqs=nseqs,
                beta=beta,
                scan=scan,
                scanrange=scanrange
            )

        biased_seqs = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased", f"genseq{seqs_fraction}.fasta")
        unbiased_seqs = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased", f"genseq{seqs_fraction}.fasta")
        biased_params = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased", f"params{seqs_fraction}.json")
        unbiased_params = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased", f"params{seqs_fraction}.json")
        null_model_seqs = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased", f"null_model{seqs_fraction}.fasta")
        natural_shorts_seq = os.path.join(os.path.join(os.path.dirname(family_dir), "shorts"), family_file.split('.')[0], f"{family_file}.fasta")
        natural_longs_seq = os.path.join(os.path.join(os.path.dirname(family_dir), "longs"), family_file.split('.')[0], f"{family_file}.fasta")
    
        if do_one:
            break
