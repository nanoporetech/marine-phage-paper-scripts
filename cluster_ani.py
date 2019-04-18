import os,sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser
import argparse

def parse_args():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("fastani", help="File containing FastANI results.", type=str)
    parser.add_argument("bin_id", help="K-mer bin ID number", type=int)

    # Optional arguments
    parser.add_argument("-p", "--prefix", help="Output files prefix [ani_clusters]", type=str, default="ani_clusters")
    parser.add_argument("-r", "--min_reads", help="Minimum number of reads in ANI cluster. Polishing with too few reads (e.g. <5) is ineffective. [10]", type=int, default=10)
    parser.add_argument("-s", "--min_sim", help="Minimum average similarity among reads in ANI cluster [0.95]", type=float, default=.95)
    parser.add_argument("-l", "--min_len", help="Minimum average read length in ANI cluster. May help filter out incomplete genome reads. [28000]", type=int, default=28000)
    parser.add_argument("-d", "--max_d", help="Cophenetic distance at which ANI cluster boundaries are drawn [1.0]", type=float, default=1.0)
    parser.add_argument("-m", "--method", help="Method used to calculate distance in hierarchical clustering [ward]", type=str, default="ward")

    # Parse arguments
    args = parser.parse_args()

    return args

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = sch.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        # plt.title('Hierarchical Clustering Dendrogram (truncated)')
        # plt.xlabel('sample index or (cluster size)')
        # plt.ylabel('distance')
        plt.xticks([])
        plt.yticks([])
        
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                # plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                #              textcoords='offset points',
                #              va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    ax = plt.gca()
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    return ddata

def create_figure(clusters, idx2, axc, heatmap_fn):
    dc       = np.array(clusters[idx2], dtype=int)
    dc.shape = (1,len(clusters))

    vals = np.linspace(0,1,256)
    np.random.shuffle(vals)
    cmap_c = plt.cm.colors.ListedColormap(plt.cm.hsv(vals))

    im_c     = axc.matshow(dc, aspect='auto', origin='lower', cmap=cmap_c)
    def find_mid_index(vec, value):
        idx_vec = np.where(vec==value)[0]
        return idx_vec[int(len(idx_vec)/2)]
    labels = dict([(value, find_mid_index(dc[0], value)) for value in set(clusters)])
    for l,x in labels.items():
        axc.text(x, 0, l, horizontalalignment='center', fontsize=12)
    axc.set_xticks([])
    axc.set_yticks([])

    plt.savefig(heatmap_fn, dpi=250)

def main(args):
    result = pd.read_csv(args.fastani, sep="\t", names=["read1_path", "read2_path", "ANI", "mappings", "fragments"])

    # strip away path and file extension from read ID
    def extract_read_id(s):
        return os.path.basename(s).replace(".fa", "")

    result["read1"] = result["read1_path"].apply(extract_read_id)
    result["read2"] = result["read2_path"].apply(extract_read_id)

    path_d = dict(zip(result["read1"], result["read1_path"]))

    def get_readlen(read):
        for header,seq in SimpleFastaParser(open(path_d[read], "r")):
            break
        return len(seq)

    # Convert FastANI results file to a pairwise ANI matrix
    result = result.pivot_table(index="read1", columns="read2", values="ANI").fillna(0.0)

    # result = result.reset_index().rename(columns={"read1": "index"}).set_index("index")
    result.loc[:,:] *= 0.01

    # Since there are really no ANI values between 0 and .75, set all zero values to 0.75
    # This will prevent the clustering algorithm from thinking that gap between 0 and 0.75 
    # is significant.
    result = result.mask(result == 0).fillna(0.75)

    # calculate full dendrogram 
    fig = plt.figure(figsize=(12, 12))

    Z   = sch.linkage(result, args.method, optimal_ordering=True)

    buff          = 0.005
    x_start       = 0.03
    y_start       = 0.23
    cb_width      = 0.05
    cb_lab_buff   = 0.03
    ld_xstart     = x_start + cb_width + cb_lab_buff + buff
    ld_width      = 0.05
    mat_xstart    = ld_xstart + ld_width + buff
    mat_width     = 0.6
    mat_height    = 0.6
    ccolor_ystart = y_start + mat_height + buff
    ccolor_h      = 0.03
    td_ystart     = y_start + mat_height + buff + ccolor_h + buff
    td_height     = 0.12

    # left side dendrogram
    ax1 = fig.add_axes([ld_xstart, y_start, ld_width, mat_height])
    ax1.set_xticks([])
    ax1.set_yticks([])
    Z1 = fancy_dendrogram(
                            Z,
                            show_contracted=True,
                            leaf_rotation=90.,  # rotates the x axis labels
                            leaf_font_size=8.,  # font size for the x axis labels
                            annotate_above=0.07,
                            orientation="left"
                        )

    # top side dendogram
    ax2 = fig.add_axes([mat_xstart, td_ystart, mat_width, td_height])
    ax2.set_xticks([])
    Z2 = fancy_dendrogram(
                            Z,
                            show_contracted=True,
                            leaf_rotation=90.,  # rotates the x axis labels
                            leaf_font_size=8.,  # font size for the x axis labels
                            annotate_above=0.07,
                            max_d=args.max_d,
                        )

    # main heat-map
    orig_cmap    = matplotlib.cm.Greys
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.90, name='shifted')
    axmatrix     = fig.add_axes([mat_xstart, y_start, mat_width, mat_height])
    idx1         = Z1['leaves']
    idx2         = Z2['leaves']
    mat_result   = result.iloc[idx1, :]
    mat_result   = mat_result.iloc[:, idx2]
    im           = axmatrix.matshow(mat_result, aspect='auto', origin='lower', cmap=shifted_cmap, vmin=0, vmax=1)

    axmatrix.set_xticks(range(len(idx1)))
    axmatrix.set_xticklabels(result.index[idx1], minor=False, fontsize=4)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    plt.xticks(rotation=-90, fontsize=4)    

    axmatrix.set_yticks(range(len(idx2)))
    axmatrix.set_yticklabels(result.index[idx2], minor=False, fontsize=4)
    axmatrix.yaxis.set_label_position('right')
    axmatrix.yaxis.tick_right()

    # colorbar
    cbaxes = fig.add_axes([x_start, y_start, cb_width, mat_height]) 
    cbar   = fig.colorbar(im, cax = cbaxes, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # top side color coding for cluster membership
    [axc_x, axc_y, axc_w, axc_h] = [mat_xstart, ccolor_ystart, mat_width, ccolor_h]
    axc      = fig.add_axes([axc_x, axc_y, axc_w, axc_h])  # axes for column side colorbar

    clusters = sch.fcluster(Z, args.max_d, criterion='distance')

    result.loc[:,"cluster"] = clusters
    result                  = result.reset_index()

    result["read1_len"]     = result["read1"].apply(get_readlen)

    cluster_dfs             = result.groupby("cluster")

    clust_info_fn = "{}.info.tsv".format(args.prefix)
    with open(clust_info_fn, "w") as f:
        f.write("bin_id,cluster,read_id,clust_read_ani,clust_mean_ani,clust_mean_len\n")
        for cluster,df in cluster_dfs:
            keep_cols        = df["read1"].append(pd.Series(["read1", "cluster", "read1_len"]))
            df               = df.loc[:,keep_cols] # only keep the columns of reads from the cluster
            read_means = []
            for idx,row in df.iterrows():
                nonself   = row.replace(1.00, np.nan).dropna()
                ani_idx   = [i for i in nonself.index if (i!="read1" and i!="cluster" and i!="read1_len")]
                read_means.append(nonself.loc[ani_idx].mean())

            df.loc[:,"ani_mean"] = read_means
            clust_mean_ani       = df.loc[:,"ani_mean"].mean()
            clust_mean_len       = df.loc[:,"read1_len"].mean()
            df = df.set_index("read1")
            if df.shape[0]>=args.min_reads and clust_mean_ani>=args.min_sim and clust_mean_len>=args.min_len:
            
                print("Bin {}, ANI cluster {}: mean ANI={} mean readlen={}".format(args.bin_id, \
                                                                                   row["cluster"], \
                                                                                   round(clust_mean_ani,4), \
                                                                                   round(clust_mean_len,1)))
                
                for r in df.index.values:
                    f.write("{},{},{},{},{},{}\n".format(args.bin_id, \
                                                         cluster, \
                                                         r, \
                                                         round(df.loc[r,"ani_mean"],4), \
                                                         round(clust_mean_ani,4), \
                                                         round(clust_mean_len,1)))

    heatmap_fn = "{}.heatmap.png".format(args.prefix)
    create_figure(clusters, idx2, axc, heatmap_fn)

if __name__=="__main__":
    args = parse_args()

    main(args)