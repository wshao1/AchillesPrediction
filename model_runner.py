import argparse
from Models import run_on_target
from multiprocessing import Pool
from data_helper import check_dir_exists_or_make


def parse_gene_list_file(gene_list_file_name):
    gene_list = []
    with open(gene_list_file_name) as f_in:
        for line in f_in:
            line = line.strip().rstrip()
            if len(line) > 0:
                gene_list.append(line)
    return gene_list


def process_gene_list(gene_effect_file_name, gene_expression_file_name, gene_list, model_name, cv_df_file_name=None,
                      num_threads=16):
    params_list = [(gene_effect_file_name, gene_expression_file_name, gene_name,
                    model_name, cv_df_file_name) for gene_name in gene_list]
    with Pool(num_threads) as p:
        res = p.starmap(run_on_target, params_list)
        p.close()
        p.join()
    return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_effect',
                        default='Achilles_gene_effect.csv')
    parser.add_argument('--gene_expression',
                        default='CCLE_expression.csv')
    parser.add_argument('--model_name', help="Options are 'linear', 'xg_boost', 'deep', 'ensemble', 'choose_best'",
                        default='choose_best')
    parser.add_argument('--cv_file', help="Cross validation ids file path. See data_helper.py for how to create such "
                                          "a file.",
                        default="cross_validation_folds_ids.tsv")
    parser.add_argument('--gene_list', help="File path of a list of gene names. Should contain one gene name per line.",
                        default="gene_file_200.txt")
    parser.add_argument('--num_threads', help="Number of threads",
                        default=1)
    parser.add_argument('--results_directory',
                        default="res/")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    gene_list = parse_gene_list_file(args.gene_list)
    check_dir_exists_or_make(args.results_directory)
    res = process_gene_list(args.gene_effect, args.gene_expression, gene_list, args.model_name, args.cv_file,
                            args.num_threads)

    x = 0