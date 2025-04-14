import os

import sample.getseq as getseq
import sample.loader as loader
import sample.kmers as kmers
import sample.analyse as viz
import sample.simcalc as simcalc

def main(interpolation_dir : str, tmp_output : str, output_dir : str, k:int, natural_path : str):
    for family in os.listdir(interpolation_dir):
        genseq_path = os.path.join(interpolation_dir, family, "biased")
        outseq_path = os.path.join(output_dir, family)

        os.makedirs(outseq_path, exist_ok=True)

        getseq.main_getseq(genseq_path=genseq_path, tmp_output=tmp_output, outseq_path=outseq_path)

    for fam, batches in loader.stream_batches(family_dir=output_dir):
        batches = kmers.encode_batches(batches=batches, k=k)
        intra_clust = simcalc.stream_batches_similarity(batches=batches)

        ordered_batches = sorted(batches, reverse=True)
        batches = {key:batches[key] for key in ordered_batches}

        simcalc.stream_batches_smaller(batches)

        pdf_file = os.path.join(output_dir, family, f"{fam}.pdf")
        pdf = viz.main_display(batches=batches, path_report=pdf_file, intra_clust=intra_clust, k=k, natural=natural_path)

        pdf.close()


if __name__ == "__main__":
    interpolation_dir = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/sequences/interpolation'
    tmp_output = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/tmp'
    output_dir = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/sampled_seq'
    natural_msa = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/data/test/Azoarcus/Azoarcus.fasta'
    k=5
    main(interpolation_dir=interpolation_dir, tmp_output=tmp_output, output_dir=output_dir, k=k, natural_path=natural_msa)