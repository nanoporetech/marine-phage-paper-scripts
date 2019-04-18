import sys
import numpy as np
from itertools import groupby
from random import choice
import argparse
from Bio.SeqIO.FastaIO import SimpleFastaParser

def parse_args():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("fasta", help="Fasta file containing the sequences to filter", type=str)
    parser.add_argument("genome_size", help="Expected genome size (bp)", type=int)

    # Optional arguments
    parser.add_argument("-p", "--prefix", help="Output file prefix [len_filtered_reads]", type=str, default="len_filtered_reads")
    parser.add_argument("-f", "--len_frac", help="Fraction of genome size to use as min length cutoff [0.9]", type=float, default=0.9)

    # Parse arguments
    args = parser.parse_args()

    return args

def main(args):
    seq_ids  = []
    seq_lens = []
    for seq_id,seq in SimpleFastaParser(open(args.fasta)):
        seq_ids.append(seq_id)
        seq_lens.append(len(seq))

    seq_lens = np.array(seq_lens)
    min_len = int(args.len_frac * args.genome_size)

    fname = "{}.fasta".format(args.prefix)
    with open(fname, "w") as out_f:
        for seq_id, seq in fasta_iter(args.fasta):
            if len(seq)>min_len:
                out_f.write(">%s\n" % seq_id)
                out_f.write("%s\n" % seq)

if __name__ == "__main__":
    args = parse_args()

    main(args)
