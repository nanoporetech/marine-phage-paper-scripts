import os,sys
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import simps
import math
import random
import argparse

# This scipt helps identify whether a viral genome size estimate
# can be obtained by simply looking for an enrichment of long reads
# of a certain length that capture the entire viral genome. These
# will appear as bumps in the read length distribution at large 
# values (i.e. >25kb) where the read lengths would otherwise be 
# expected to trail off significantly.

def parse_args():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("hdbscan_bins", help="File containing the bin assignments for all reads.", type=str)

    # Optional arguments
    parser.add_argument("-p", "--prefix", help="Output file prefix [bin_filter]", type=str, default="bin_filter")
    parser.add_argument("--length_min", help="Left bound of the read length distribution to consider [20000]", type=int, default=20000)
    parser.add_argument("--length_max", help="Right bound of the read length distribution to consider [80000]", type=int, default=80000)
    parser.add_argument("-w", "--min_window_area", help="Min fraction of AUC for a slice area to be considered a peak (changing not recommended) [0.04]", type=float, default=0.04)

    # Parse arguments
    args = parser.parse_args()

    return args

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def main(args):
    df              = pd.read_csv(args.hdbscan_bins, sep="\t")

    # Number of discrete read length values to use for building model of readlength distribution
    n_points    = 400
    # Number of points in sliding window to use for measuring window AUC
    window_size = 10
    # Maximum number of plot on each figure 
    max_fig_axes = 64

    # Sample 1000 reads from all reads in 2D map
    np.random.seed(1)
    rl_sample       = np.random.choice(df["length"], 1000, replace=False)
    all_kde         = stats.kde.gaussian_kde( rl_sample )
    orig_dist_space = np.linspace( args.length_min, args.length_max, n_points )

    # Only look at readlengths longer than the most common total readlength
    # dist_space = orig_dist_space[all_kde(orig_dist_space).argmax():]
    dist_space = orig_dist_space

    for_out_df = []
    bin_ids = df["bin_id"].unique()
    for bin_id in bin_ids:
        signif = False
        bin_readlens = df[df["bin_id"]==bin_id]["length"]
        bin_kde      = stats.kde.gaussian_kde(bin_readlens)

        all_vals  = all_kde(dist_space)
        bin_vals  = bin_kde(dist_space)
        diff_vals = bin_vals - all_vals
        diff_vals = np.clip( diff_vals, 0, 80000)
        all_area  = simps(diff_vals, dx=n_points)
        
        # Slide through the readlens, calculating area for each window
        peak_readlens = []
        for i in range(len(dist_space)):
            dist_window = dist_space[i:(i+window_size)]
            window_vals = diff_vals[i:(i+window_size)]

            window_area = simps(window_vals, dx=n_points)
            # Check if interval area contains a sufficient fraction of total AUC to be considered a "peak"
            if window_area>args.min_window_area:
                
                # keep track of where the peak occurred in this interval
                peak_idx = np.argmax(window_vals)
                peak_readlens.append( (dist_window[peak_idx], window_vals[peak_idx]) )

        if len(peak_readlens)>0:
            X = np.asarray(peak_readlens)
            genomesize = X[X[:,1].argmax(), 0]

            # Need to be able to separate from background readlength distribution (> 110% of peak value in all reads)
            if genomesize > (1.1*dist_space[0]):
                signif = True
            else:
                genomesize = 0
        else:
            genomesize = 0

        for_out_df += list(zip([bin_id]*len(dist_space), \
                                dist_space, \
                                all_vals, \
                                bin_vals, \
                                diff_vals, \
                                [signif]*len(dist_space), \
                                [genomesize]*len(dist_space)))

    bins_df = pd.DataFrame(for_out_df, columns=["bin_id", "length_coord", "all_model", "bin_model", "model_diff", "significant", "genomesize_est"])

    stats_df = bins_df.groupby(["bin_id", "genomesize_est"]).size().reset_index()
    stats_df["rl_peak"] = stats_df["genomesize_est"]>0
    
    print("Found {} of {} bins with significant spikes in read length distribution...".format(stats_df[stats_df["rl_peak"]==True].shape[0], stats_df.shape[0]))
    fname = "{}.bin_rl_filter.tsv".format(args.prefix)
    print(fname)
    stats_df["genomesize_est"] = stats_df["genomesize_est"].astype(int)
    stats_df[["bin_id", "rl_peak", "genomesize_est"]].to_csv(fname, sep="\t", index=False)

    print()
    sigbin_fig_tracker = {}
    sig_bin_ids        = bins_df[bins_df["significant"]==True]["bin_id"].unique()
    sig_bin_ids.sort()
    sig_bin_ids_chunks = list(chunks(sig_bin_ids, max_fig_axes))
    # dictionary where key=fig_id and value=(fig, axes)
    sig_fig_axes       = {}
    for i,sig_ids_chunk in enumerate(sig_bin_ids_chunks):
        # remember which figure each sig_bin_id maps to
        for bin_id in sig_ids_chunk:
            sigbin_fig_tracker[bin_id] = i
        sig_grid_n     = math.ceil(math.sqrt(len(sig_ids_chunk)))
        sig_fig, axes1 = plt.subplots(nrows=sig_grid_n, ncols=sig_grid_n, sharey=False, figsize=[20,20])
        if len(sig_ids_chunk)==1:
            sig_fig_axes[i] = (sig_fig, [axes1])
        else:
            sig_fig_axes[i] = (sig_fig, [item for sublist in axes1 for item in sublist])

    nonsigbin_fig_tracker = {}
    nonsig_bin_ids        = bins_df[bins_df["significant"]==False]["bin_id"].unique()
    nonsig_bin_ids.sort()
    nonsig_bin_ids_chunks = list(chunks(nonsig_bin_ids, max_fig_axes))
    nonsig_fig_axes       = {}
    for i,nonsig_ids_chunk in enumerate(nonsig_bin_ids_chunks):
        # remember which figure each nonsig_bin_id maps to
        for bin_id in nonsig_ids_chunk:
            nonsigbin_fig_tracker[bin_id] = i
        nonsig_grid_n     = math.ceil(math.sqrt(len(nonsig_ids_chunk)))
        nonsig_fig, axes2 = plt.subplots(nrows=nonsig_grid_n, ncols=nonsig_grid_n, sharey=False, figsize=[20,20])
        if len(nonsig_ids_chunk)==1:
            nonsig_fig_axes[i] = (nonsig_fig, [axes2])
        else:
            nonsig_fig_axes[i] = (nonsig_fig, [item for sublist in axes2 for item in sublist])

    for bin_id,df_ in bins_df.groupby("bin_id"):
        if bin_id in sigbin_fig_tracker.keys():
            fig_id  = sigbin_fig_tracker[bin_id]
            sig_fig = sig_fig_axes[fig_id][0]
            axes    = sig_fig_axes[fig_id][1]
            ax      = axes.pop(0)
        else:
            fig_id     = nonsigbin_fig_tracker[bin_id]
            nonsig_fig = nonsig_fig_axes[fig_id][0]
            axes       = nonsig_fig_axes[fig_id][1]
            ax         = axes.pop(0)
        
        df_.plot(x="length_coord", y="all_model",  ax=ax, label="all reads", linestyle="--", color="k")
        df_.plot(x="length_coord", y="bin_model",  ax=ax, label="bin reads", linestyle="--", color="b")
        df_.plot(x="length_coord", y="model_diff", ax=ax, label="pdf diff",  linestyle="--", color="r")
        ax.fill_between(df_["length_coord"], 0, df_["model_diff"], color="r")
        ax.legend()    
       
        # ax.grid(linewidth=0.25)
        ax.set_title("kmer bin {}".format(bin_id))
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    print("Plotting bin read length distributions...")
    for fig_id,(sig_fig, axes_list) in sig_fig_axes.items():
        fname = "{}.peaks.{}.png".format(args.prefix, fig_id)
        print(fname)
        for ax in axes_list:
            ax.remove()
        sig_fig.tight_layout()
        sig_fig.savefig(fname, dpi=150)

    for fig_id,(nonsig_fig, axes_list) in nonsig_fig_axes.items():
        fname = "{}.nopeaks.{}.png".format(args.prefix, fig_id)
        print(fname)
        for ax in axes_list:
            ax.remove()
        nonsig_fig.tight_layout()
        nonsig_fig.savefig(fname, dpi=150)

if __name__=="__main__":
    args = parse_args()

    main(args)