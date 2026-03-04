#-------------------------------------------------------------------------------
# Name:        mtype - Gene Set Similarity Analysis
# Purpose:     Compute hypergeometric p-values and similarity metrics
#
# Author:      yzlco
#
# Created:     21/11/2019
# Modified:    2025-01 - Added pre-filtered input support and parameterized weights
# Copyright:   (c) yzlco 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import pandas
import codecs
import numpy
import csv
import argparse
from scipy.stats import hypergeom
from multiprocessing import Process, Lock, Value, Pool
import multiprocessing
import threading

# Default composite weight (α for Jaccard)
DEFAULT_ALPHA = 0.8

class myDict:
    def __init__(self):
        self._dict = {}
    def add(self, id, val):
        self._dict[id] = val
    def getval(self):
        return self._dict.values()
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out
def compute_similarity(ovl, s1N, s2N, alpha=DEFAULT_ALPHA):
    """Compute weighted similarity from overlap counts.

    Args:
        ovl: Overlap count
        s1N: Size of set 1
        s2N: Size of set 2
        alpha: Weight for Jaccard (default 0.8)

    Returns:
        Weighted similarity score
    """
    simpson = ovl / numpy.min([s1N, s2N]) if numpy.min([s1N, s2N]) > 0 else 0
    jaccard = ovl / (s1N + s2N - ovl) if (s1N + s2N - ovl) > 0 else 0
    return alpha * jaccard + (1 - alpha) * simpson


def mtype(indexA, indexB, lock, PAGLex, g, pmf, TotalGene, alpha=DEFAULT_ALPHA):
    a = g.get_group(PAGLex[indexA])
    b = g.get_group(PAGLex[indexB])
    s1=a['GENE_SYMBOL']
    s2=b['GENE_SYMBOL']
    #https://stackoverflow.com/questions/20192968/how-to-measure-overlap-of-groups-in-pandas-groupby-objects
    #intersect=pandas.Series(list(set(a['GENE_SYM']).intersection(set(b['GENE_SYM']))))
    s1N=s1.count()
    s2N=s2.count()
    ovl=s1[s1.isin(s2)].count()
    similarity = compute_similarity(ovl, s1N, s2N, alpha)
    lock.acquire()
    if(TotalGene*ovl <= s1N*s2N):
        s1_s2_nlogpmf=0
        s1_s2_cdf=hypergeom.cdf(ovl-1, TotalGene, s1N, s2N)
    else:
        s1_s2_nlogpmf=-hypergeom.logpmf(ovl, TotalGene, s1N, s2N)
        s1_s2_cdf=hypergeom.cdf(ovl-1, TotalGene, s1N, s2N) #
        s1_s2_nlogcdf=-hypergeom.logcdf(numpy.min([s1N,s2N])-ovl, TotalGene, s1N, s2N)
        pmf.append([PAGLex[indexA],PAGLex[indexB],TotalGene,s1N, s2N, ovl, similarity, s1_s2_nlogpmf,s1_s2_cdf,s1_s2_nlogcdf])
    lock.release()

def extractPMFindexRange(indexA, indexBRange, lock, PAGLex, g, pmf, TotalGene, alpha=DEFAULT_ALPHA):
    #
    for indexB in indexBRange:
        #print(indexB)
        mtype(indexA, indexB, lock, PAGLex, g, pmf, TotalGene, alpha)




def writeMtypeToFile(filename,pmf):
    outfile = open(str(filename)+'.txt', 'w')
    outfile.write("GS_A_ID\tGS_B_ID\tALL\tGS_A_SIZE\tGS_B_SIZE\tOLAP\tSIMILARITY\tNLOGPMF\tCDF\tNLOGCDF\n")
    for data_slice in pmf:
        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.
        writeline = ""
        for ele in data_slice:
            writeline+=str(ele) + '\t'
        #print(writeline[:-1])
        outfile.write(writeline[:-1])
        # Writing out a break to indicate different slices...
        outfile.write('\n')
    outfile.close()


def load_prefiltered_pairs(candidates_file):
    """Load gene set pairs from a pre-filtered candidates file.

    Args:
        candidates_file: Path to CSV file with GS_ID_A and GS_ID_B columns

    Returns:
        List of (gs_id_a, gs_id_b) tuples
    """
    pairs = []
    with open(candidates_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gs_a = row.get('GS_ID_A', row.get('gs_id_a', ''))
            gs_b = row.get('GS_ID_B', row.get('gs_id_b', ''))
            if gs_a and gs_b:
                pairs.append((gs_a, gs_b))
    return pairs


def process_prefiltered(candidates_file, gene_set_file='GO2gene_aggregate.txt',
                        output_file='mtype_prefiltered', alpha=DEFAULT_ALPHA,
                        maxthread=6):
    """Process pre-filtered candidate pairs with hypergeometric statistics.

    This is Stage 2 of the two-stage filtering pipeline.
    Only computes expensive statistics for pre-filtered candidates.

    Args:
        candidates_file: Path to pre-filtered candidates CSV
        gene_set_file: Path to gene set data file
        output_file: Output file name (without extension)
        alpha: Composite weight for similarity
        maxthread: Number of parallel threads
    """
    print(f"Loading pre-filtered candidates from {candidates_file}...")
    pairs = load_prefiltered_pairs(candidates_file)
    print(f"  Loaded {len(pairs):,} candidate pairs")

    # Build lookup of pair indices
    pair_set = set(pairs)

    # Load gene set data
    print(f"Loading gene sets from {gene_set_file}...")
    PAGgn = pandas.read_csv(gene_set_file, sep=',', index_col=False, encoding='unicode_escape')
    PAGgnUni = PAGgn.drop_duplicates(subset=None, keep='first', inplace=False)

    # Filter empty gene sets
    g = PAGgnUni.groupby("GS_ID")
    gene_set_sizes = g.size()
    non_empty_gs = gene_set_sizes[gene_set_sizes > 0].index.tolist()
    print(f"  Found {len(non_empty_gs):,} non-empty gene sets")

    TotalGene = len(PAGgnUni.groupby("GENE_SYMBOL").count())
    print(f"  Total unique genes: {TotalGene:,}")

    # Get unique gene sets from pairs
    unique_gs = set()
    for gs_a, gs_b in pairs:
        unique_gs.add(gs_a)
        unique_gs.add(gs_b)

    # Filter to valid gene sets
    valid_gs = [gs for gs in unique_gs if gs in g.groups]
    missing_gs = unique_gs - set(valid_gs)
    if missing_gs:
        print(f"  WARNING: {len(missing_gs)} gene sets not found in data")

    print(f"\nProcessing {len(pairs):,} pairs...")

    manager = multiprocessing.Manager()
    pmf = manager.list()
    lock = Lock()

    # Process each pair
    processed = 0
    for gs_a, gs_b in pairs:
        if gs_a not in g.groups or gs_b not in g.groups:
            continue

        a = g.get_group(gs_a)
        b = g.get_group(gs_b)
        s1 = a['GENE_SYMBOL']
        s2 = b['GENE_SYMBOL']

        s1N = s1.count()
        s2N = s2.count()
        ovl = s1[s1.isin(s2)].count()
        similarity = compute_similarity(ovl, s1N, s2N, alpha)

        if TotalGene * ovl <= s1N * s2N:
            s1_s2_nlogpmf = 0
            s1_s2_cdf = hypergeom.cdf(ovl - 1, TotalGene, s1N, s2N)
        else:
            s1_s2_nlogpmf = -hypergeom.logpmf(ovl, TotalGene, s1N, s2N)
            s1_s2_cdf = hypergeom.cdf(ovl - 1, TotalGene, s1N, s2N)
            s1_s2_nlogcdf = -hypergeom.logcdf(numpy.min([s1N, s2N]) - ovl, TotalGene, s1N, s2N)
            pmf.append([gs_a, gs_b, TotalGene, s1N, s2N, ovl, similarity,
                       s1_s2_nlogpmf, s1_s2_cdf, s1_s2_nlogcdf])

        processed += 1
        if processed % 1000 == 0:
            print(f"  Processed {processed:,}/{len(pairs):,} pairs", flush=True)

    print(f"\n  Writing {len(pmf)} results to {output_file}.txt")
    writeMtypeToFile(output_file, pmf)
    print("  Done.")

    return list(pmf)

class PAGERmtype():
    def __new__(self,pagIDs):
        # pubmed/PAGER/input/PAG_Gene.txt
        PAGgn = pandas.read_csv('GO2gene_aggregate.txt', sep='\t', index_col=False,encoding = 'unicode_escape')
        PAGgnUni = PAGgn.drop_duplicates(subset=None, keep='first', inplace=False)

        TotalGene=len(PAGgnUni.groupby("GENE_SYMBOL").count())
        #PAGLex=PAGgnUni['GS_ID'].unique().tolist()
        # limit to query IDs
        PAGLex=pagIDs
        g = PAGgnUni.groupby("GS_ID")

        maxthread = 6
        manager = multiprocessing.Manager()
        pmf = manager.list()
        lock = Lock()
        for indexA in range(0,len(PAGLex)-1):
        #for indexA in range(0,1):
            if(indexA%1==0):
                print("A:%s out of %s" %(indexA+1,len(PAGLex)))
                start = indexA+1
                if(len(PAGLex)-indexA-1<maxthread):
                    chunk = chunkIt(range(indexA+1,len(PAGLex)),len(PAGLex)-indexA-1)
                else:
                    chunk = chunkIt(range(indexA+1,len(PAGLex)),maxthread)
                threads = []
                for i in range(0, len(chunk)):#0-400
                    edgeIindexRange = chunk[i]
                    print("start:%s-%s" %(min(edgeIindexRange), max(edgeIindexRange)))
                    t=Process(target=extractPMFindexRange, args=[indexA, edgeIindexRange,lock,PAGLex,g,pmf,TotalGene])
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()
        return(pmf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gene set similarity analysis with hypergeometric statistics'
    )
    parser.add_argument('--prefiltered', '-p', default=None,
                        help='Path to pre-filtered candidates CSV (Stage 2 mode)')
    parser.add_argument('--gene-sets', '-g', default='GO2gene_aggregate.txt',
                        help='Gene set file (default: GO2gene_aggregate.txt)')
    parser.add_argument('--output', '-o', default='m_type_hybrid',
                        help='Output file name without extension (default: m_type_hybrid)')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help=f'Composite weight for Jaccard (default: {DEFAULT_ALPHA})')
    parser.add_argument('--threads', '-t', type=int, default=6,
                        help='Number of threads (default: 6)')

    args = parser.parse_args()

    if args.prefiltered:
        # Stage 2: Process pre-filtered candidates only
        print("=" * 60)
        print("Stage 2: Processing Pre-filtered Candidates")
        print("=" * 60)
        process_prefiltered(
            candidates_file=args.prefiltered,
            gene_set_file=args.gene_sets,
            output_file=args.output,
            alpha=args.alpha,
            maxthread=args.threads
        )
    else:
        # Original behavior: process all pairs
        print("=" * 60)
        print("Processing All Gene Set Pairs")
        print("=" * 60)
        # freeze_support() here if program needs to be frozen
        PAGgn = pandas.read_csv(args.gene_sets, sep=',', index_col=False, encoding='unicode_escape')

        PAGgnUni = PAGgn.drop_duplicates(subset=None, keep='first', inplace=False)
        TotalGene = len(PAGgnUni.groupby("GENE_SYMBOL").count())
        PAGLex = PAGgnUni['GS_ID'].unique().tolist()
        g = PAGgnUni.groupby("GS_ID")
        maxthread = args.threads
        manager = multiprocessing.Manager()
        pmf = manager.list()
        lock = Lock()
        for indexA in range(0, len(PAGLex) - 1):
            if indexA % 1 == 0:
                print("A:%s out of %s" % (indexA + 1, len(PAGLex)))
                start = indexA + 1
                if len(PAGLex) - indexA - 1 < maxthread:
                    chunk = chunkIt(range(indexA + 1, len(PAGLex)), len(PAGLex) - indexA - 1)
                else:
                    chunk = chunkIt(range(indexA + 1, len(PAGLex)), maxthread)
                threads = []
                for i in range(0, len(chunk)):
                    edgeIindexRange = chunk[i]
                    t = Process(target=extractPMFindexRange,
                               args=[indexA, edgeIindexRange, lock, PAGLex, g, pmf, TotalGene, args.alpha])
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()
        writeMtypeToFile(args.output, pmf)
