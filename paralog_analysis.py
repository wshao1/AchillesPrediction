import os
import gget
import numpy as np
from matplotlib import pyplot as plt
from random import sample


def plot_enrich(annotations, p_val_list, t_name):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    # annotations = ['Metabolism of xenobiotics by cytochrome P450 *1.57e-11', 'Chemical carcinogenesis *1.73e-9', 'Ascorbate and aldarate metabolism *7.5e-9', 'Steroid hormone biosynthesis *1.03e-8', 'Drug metabolism *1.20e-8']
    annotations.reverse()
    y_pos = np.arange(len(annotations))
    # performance = [1.57e-11, 1.73e-09, 7.54e-09, 1.03e-08, 1.20e-08]
    p_val_list = [-np.log10(x) for x in p_val_list]
    p_val_list.reverse()

    bar_plot = ax.barh(y_pos, p_val_list, align='center')
    # ax.set_yticks(y_pos)
    # ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('-Log10(p-value)')
    ax.set_title(f'{t_name}')
    ax.yaxis.set_ticks([])


    def autolabel(bar_plot):
        for idx, rect in enumerate(bar_plot):
            ax.text(0.25, idx + .10, annotations[idx], verticalalignment='top', color='white', size=11)
    autolabel(bar_plot)
    fig.tight_layout()
    plt.show()


def enrichment_analysis():
    features_dir = "features_dir"
    num_trials = 0
    successes = 0
    for f in os.listdir(features_dir):
        if f.endswith("features.txt"):
            # if not "CHEK2" in f: #CHEK2 TP53 MDM2 USP28 WRAP73
            #     continue
            target_gene_name = f.split(".")[0]
            if target_gene_name != 'WRAP73':
                continue
            gene_list = []
            with open(os.path.join(features_dir, f), "r") as f_in:
                for line in f_in:
                    line = line.strip()
                    gene_list.append(line)
            res = gget.enrichr(gene_list, database="pathway")
            if target_gene_name == 'WRAP73':
                pathway_names = list(res.path_name)[:5]
                p_vals = list(res.p_val)[:5]
                plot_enrich(pathway_names, p_vals, target_gene_name)
            adjusted_p_val = res.values[0][6]
            print(f"target {target_gene_name} adjusted p: {adjusted_p_val}")
            if adjusted_p_val < 0.05:
                successes += 1
            num_trials += 1
    print(f"success: {successes} trials: {num_trials}")


def permutation_test(num_tests, paralog_dict, all_genes, real_count):
    num_better = 0
    all_genes_arr = np.array(all_genes)
    for i in range(num_tests):
        contains_paralog = 0
        total_genes = len(all_genes)
        indices = list(range(total_genes))
        total = 0
        for file in os.listdir(features_dir):
            if file.endswith("features.txt"):
                target_gene = file.split(".")[0]
                features_list = []
                if target_gene in paralog_dict:
                    total += 1
                    num_modifiers = 0
                    with open(os.path.join(features_dir, file), "r") as feature_f:
                        for line in feature_f:
                            if len(line.strip()) > 0:
                                num_modifiers += 1
                    paralogs = paralog_dict[target_gene]
                    cur_indices = sample(indices, num_modifiers)
                    cur_genes = all_genes_arr[cur_indices]
                    for g in cur_genes:
                        if g in paralogs:
                            contains_paralog += 1
                            break
        if contains_paralog > real_count:
            num_better += 1
    print(f"{num_better/num_tests}")
    return num_better, num_tests


if __name__ == '__main__':
    enrichment_analysis()
    file_name = "/cs/cbio/jon/projects/PyCharmProjects/AchillesPrediction/paralogs.txt"
    all_genes = []
    all_gene_names = "/cs/cbio/jon/projects/PyCharmProjects/AchillesPrediction/all_gene_names.txt"
    with open(all_gene_names, "r") as f:
        for line in f:
            tokens = line.strip().split("(")
            if len(tokens) > 1:
                all_genes.append(tokens[0][:-1])
    paralog_dict = {}
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if line == "hgnc":
                continue
            genes = [x.split("->")[0] for x in line.split(" ")]
            if len(genes) > 1:
                for g in genes:
                    if g != "NaN":
                        paralog_dict[g] = set(genes)
    features_dir = "/cs/cbio/jon/projects/PyCharmProjects/AchillesPrediction/features_dir"
    contains_paralog = 0
    total = 0
    for file in os.listdir(features_dir):
        if file.endswith("features.txt"):
            target_gene = file.split(".")[0]
            features_list = []
            if target_gene in paralog_dict:
                paralogs = paralog_dict[target_gene]
                total += 1
                with open(os.path.join(features_dir, file), "r") as feature_f:
                    for line in feature_f:
                        cur_feature = line.strip()
                        if cur_feature != target_gene and cur_feature in paralogs:
                            contains_paralog += 1
                            break
    permutation_test(100, paralog_dict, all_genes, contains_paralog)
    x = 0