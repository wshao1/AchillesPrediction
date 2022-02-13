import argparse
from multiprocessing import Pool
from data_helper import check_dir_exists_or_make
from essentiality_predictor import predict_all_tissues


def parse_gene_list_file(gene_list_file_name):
    gene_list = []
    with open(gene_list_file_name) as f_in:
        for line in f_in:
            line = line.split("(")[0].strip().rstrip()
            if len(line) > 0:
                gene_list.append(line)
    return gene_list


def process_gene_list(sample_file_name, gene_list, model_name,
                      num_threads=16, log_output=None):
    params_list = [(sample_file_name, gene_name, model_name) for gene_name in gene_list]
    with Pool(num_threads) as p:
        res = p.starmap(predict_all_tissues, params_list)
        p.close()
        p.join()
    return res


def print_results(gene_results, out_file):
    first_tissue_names = list(zip(*gene_results[0][0]))[0]
    with open(out_file, 'w') as f_out:
        to_print_str = "\t".join(list(first_tissue_names)) + "\n"
        f_out.write("Gene_name" + "\t" + to_print_str)
        for cur_zipped_list in gene_results:
            gene_name = cur_zipped_list[1]
            unzipped_list = list(zip(*cur_zipped_list[0]))
            tissue_names = unzipped_list[0]
            preds = unzipped_list[1]
            assert(tissue_names == first_tissue_names)
            to_print_str = "\t".join(list(["{:.4f}".format(round(x, 4)) for x in preds])) + "\n"
            f_out.write(gene_name + "\t" + to_print_str)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_file_name',
                        default='gene_expression_tissue.descartes.csv')
    parser.add_argument('--model_name', help="Options are 'linear', 'xg_boost', 'deep', 'ensemble', 'choose_best'",
                        default='xg_boost')
    parser.add_argument('--gene_list', help="File path of a list of gene names. Should contain one gene name per line.",
                        default="gene_files/gene_file_200.txt")
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
    res = process_gene_list(args.sample_file_name, gene_list, args.model_name,
                            args.num_threads, args.log_output)
    # out_file = args.results_directory + gene_list_name + "_{}_50_neighbors".format(args.model_name) + ".predictions.txt"
    # check_dir_exists_or_make(args.results_directory)
    # print_results(res, out_file)
