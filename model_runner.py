#!/usr/bin/env python3
import argparse
import os
from Models import run_on_target, choose_features
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
                      num_folds=1,
                      train_test_file_name=None,
                      num_threads=16, log_output=None, num_features=20):
    params_list = [(gene_effect_file_name, gene_expression_file_name, gene_name,
                    model_name, log_output, None, int(num_folds), cv_df_file_name, train_test_file_name, True, num_features) for gene_name in gene_list]
    with Pool(num_threads) as p:
        res = p.starmap(run_on_target, params_list)
        p.close()
        p.join()
    return res


def process_gene_list_features(gene_effect_file_name, gene_expression_file_name, gene_list,
                      num_folds=1,
                      train_test_file_name=None,
                      num_threads=16, log_output=None, num_features=20):
    params_list = [(gene_effect_file_name, gene_expression_file_name, gene_name,
                    log_output, int(num_folds), train_test_file_name, num_features) for gene_name in gene_list]
    with Pool(num_threads) as p:
        res = p.starmap(choose_features, params_list)
        p.close()
        p.join()
    return res


def print_results(gene_results, out_file):
    with open(out_file, 'w') as f_out:
        for gene_name, rmse, corr, p_val, model, _, _ in gene_results:
            if model is not None:
                to_write = "{}\t{}\t{}\t{}\t{}\n".format(gene_name, str(rmse), str(corr), str(p_val), model)
            else:
                continue
                to_write = "{}\t{}\t{}\n".format(gene_name, str(rmse), str(corr))
            f_out.write(to_write)


def print_results_features(gene_results, out_file):
    with open(out_file, 'w') as f_out:
        for target_gene, feature_list in gene_results:
            if len(feature_list) > 0:
                f_out.write(f"{target_gene}\t")
                feature_list.reverse()
                for importance, gene_name in feature_list:
                    f_out.write(f"{gene_name},{importance}\t")
                f_out.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_effect_file',
                        default='Achilles_gene_effect.csv')
    parser.add_argument('--gene_expression_file',
                        default='CCLE_expression.csv')
    parser.add_argument('--model_name', help="Options are 'linear', 'xg_boost', 'deep', 'ensemble', 'choose_best'",
                        default='choose_best')
    parser.add_argument('--cv_file', help="Cross validation ids file path. See data_helper.py for how to create such "
                                          "a file.",
                        default="cross_validation_folds_ids.tsv")
    parser.add_argument('--num_folds', help="Cross validation folds. Default is train/test, i.e. 1",
                        default=1)
    parser.add_argument('--train_test_file', help="train/test ids file path. See data_helper.py for how to create"
                                                  " such a file.",
                        default="train_test_split.tsv")
    parser.add_argument('--gene_list', help="File path of a list of gene names. Should contain one gene name per line.",
                        default="gene_files/gene_file_small.txt")
    parser.add_argument('--num_threads', help="Number of threads",
                        default=16)
    parser.add_argument('--features_mode', action='store_true')
    parser.add_argument('--results_directory',
                        default="esti_genes/")
    parser.add_argument('--num_features',
                        default=20)
    parser.add_argument('--log_output', help="A filename. default output is to std.out",
                        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    gene_list_name = args.gene_list.split("/")[-1].split(".")[0]
    gene_list = parse_gene_list_file(args.gene_list)
    if not args.features_mode:
        res = process_gene_list(args.gene_effect_file, args.gene_expression_file, gene_list, args.model_name, args.cv_file,
                                args.num_folds,
                                args.train_test_file,
                                int(args.num_threads), args.log_output, int(args.num_features))
        out_file = os.path.join(args.results_directory, gene_list_name + "_{}_".format(args.model_name) + ".res.txt")
        check_dir_exists_or_make(args.results_directory)
        print_results(res, out_file)
    else:
        res = process_gene_list_features(args.gene_effect_file, args.gene_expression_file, gene_list,
                                args.num_folds,
                                args.train_test_file,
                                int(args.num_threads), args.log_output, int(args.num_features))
        out_file = os.path.join(args.results_directory, gene_list_name + "_{}_".format(args.model_name) + ".res.txt")
        check_dir_exists_or_make(args.results_directory)
        print_results_features(res, out_file)
