import argparse
from Models import run_on_target
from multiprocessing import Pool
from data_helper import check_dir_exists_or_make


def parse_gene_list_file(gene_list_file_name):
    gene_list = []
    with open(gene_list_file_name) as f_in:
        for line in f_in:
            line = line.split("(")[0].strip().rstrip()
            if len(line) > 0:
                gene_list.append(line)
    return gene_list


def process_gene_list(gene_effect_file_name, gene_expression_file_name, gene_list, model_name, cv_df_file_name=None,
                      num_threads=16, log_output=None):
    params_list = [(gene_effect_file_name, gene_expression_file_name, gene_name,
                    model_name, log_output, cv_df_file_name) for gene_name in gene_list]
    with Pool(num_threads) as p:
        res = p.starmap(run_on_target, params_list)
        p.close()
        p.join()
    return res


def print_results(gene_results, out_file):
    with open(out_file, 'w') as f_out:
        for gene_name, rmse, corr in gene_results:
            to_write = "{}\t{}\t{}\n".format(gene_name, str(rmse), str(corr))
            f_out.write(to_write)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_effect_file',
                        default='Achilles_gene_effect.csv')
    parser.add_argument('--gene_expression_file',
                        default='CCLE_expression.csv')
    parser.add_argument('--model_name', help="Options are 'linear', 'xg_boost', 'deep', 'ensemble', 'choose_best'",
                        default='xg_boost')
    parser.add_argument('--cv_file', help="Cross validation ids file path. See data_helper.py for how to create such "
                                          "a file.",
                        default="cross_validation_folds_ids.tsv")
    parser.add_argument('--gene_list', help="File path of a list of gene names. Should contain one gene name per line.",
                        default="gene_files/gene_file_0.txt")
    parser.add_argument('--num_threads', help="Number of threads",
                        default=16)
    parser.add_argument('--results_directory',
                        default="res/")
    parser.add_argument('--log_output', help="A filename. default output is to std.out",
                        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    gene_list_name = args.gene_list.split("/")[-1].split(".")[0]
    gene_list = parse_gene_list_file(args.gene_list)
    res = process_gene_list(args.gene_effect_file, args.gene_expression_file, gene_list, args.model_name, args.cv_file,
                            args.num_threads, args.log_output)
    out_file = args.results_directory + gene_list_name + ".res.txt"
    check_dir_exists_or_make(args.results_directory)
    print_results(res, out_file)
