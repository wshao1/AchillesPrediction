import argparse
import os
import gzip as gz

from statsmodels.stats.multitest import multipletests


def chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def choose_blocks_by_fdr_bh(pvals, labels, alpha=0.05):
    rejected_list, corrected_p_vals, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    index = 0
    for rejected in rejected_list:
        if not rejected:
            break
        index += 1
    if index > 0:
        return pvals[0:index], labels[0:index]
    else:
        return [], []


def process_res_file(in_file):
    gene_p_val_list = []
    seen_genes = set()
    with gz.open(in_file, 'r') as f:
        for line in f:
            line = line.decode().strip()
            tokens = line.split("\t")
            gene_name = tokens[0]
            p_val = tokens[3]
            if gene_name not in seen_genes:
                gene_p_val_list.append((gene_name, float(p_val), float(tokens[2])))
                seen_genes.add(gene_name)

    gene_p_val_list.sort(key=lambda x: x[1])
    gene_names = [a for a, b, _ in gene_p_val_list]
    p_vals = [b for a, b, _ in gene_p_val_list]
    correlations = [c for a,b,c in gene_p_val_list]
    res_p_vals, res_gene_names = choose_blocks_by_fdr_bh(p_vals, gene_names)
    print(res_gene_names)
    print(len(res_gene_names))
    out_dir = "/cs/cbio/jon/projects/PyCharmProjects/AchillesPrediction/gene_files_significant"
    gene_chunks = chunks(res_gene_names, 17)
    file_idx = 0
    for chunk in gene_chunks:
        out_file = os.path.join(out_dir, f"gene_file_{file_idx}.txt")
        with open(out_file, 'w') as f_out:
            for g in chunk:
                f_out.write(f"{g}\n")
        file_idx += 1



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_file',
                        default='res/all_genes.txt.gz')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_res_file(args.res_file)