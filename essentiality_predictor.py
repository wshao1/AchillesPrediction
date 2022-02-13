import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from Models import run_on_target, EnsembleIDConverter
from configuration import gene_effect_file, gene_expression_file, ensemble_id_name_map_file


def get_sample_genes(rna_seq_df, ens_id_col_name=None):
    if ens_id_col_name is None:
        sample_gene_id_column_name = "RowID"
    else:
        sample_gene_id_column_name = ens_id_col_name
    converter = EnsembleIDConverter()
    rna_seq_df[converter.gene_column_name] = rna_seq_df[sample_gene_id_column_name].apply(lambda x: converter.gene_id_to_name_lookup(x))
    rna_seq_df = rna_seq_df[rna_seq_df[converter.gene_column_name] != "not-found"]
    return set(rna_seq_df[converter.gene_column_name]), rna_seq_df


def create_prediction_model(sample_file_name, target_gene, model=None, model_name='GP'):
    sep = ","
    if sample_file_name.endswith("tsv"):
        sep = "\t"
    rna_seq = pd.read_csv(sample_file_name, sep=sep)
    in_use_genes, rna_seq = get_sample_genes(rna_seq)
    if model is None:
        _, _, _, _, model, features = run_on_target(gene_effect_file, gene_expression_file, target_gene,
                                                    model_name, None, descartes_data=rna_seq, num_folds=0, return_model=True,
                                                    genes_for_features=in_use_genes, use_knn=True)
    rna_seq = rna_seq.loc[rna_seq[EnsembleIDConverter().gene_column_name].isin(set(features))]
    gene_name_col = EnsembleIDConverter().gene_column_name
    rna_seq = rna_seq.sort_values(by=[gene_name_col])
    to_drop_ids = []
    seen_names = set()
    names = list(rna_seq[gene_name_col])
    ids = list(rna_seq[rna_seq.columns[0]])
    for cur_name, row_id in zip(names, ids):
        if cur_name in seen_names:
            to_drop_ids.append(row_id)
        else:
            seen_names.add(cur_name)
    rna_seq = rna_seq.loc[~rna_seq[rna_seq.columns[0]].isin(set(to_drop_ids))]
    return rna_seq, model


def predict_for_target_tissue(rna_seq_df, tissue_name, model):
    for_prediction = rna_seq_df[tissue_name]
    as_array = np.array(for_prediction)
    # m = 0.235969463216031
    # b = 13.05616897327613
    # as_array = (as_array) / m
    x = np.array([np.log2(as_array + 1)])
    # x = model.enrich_with_knn(x)
    # return model.predict_with_std(x)
    return model.predict(x)


def predict_all_tissues(sample_file_name, target_gene, model_name='GP', model=None):
    try:
        out_file = f'/cs/zbio/jrosensk/essentiality_predictions/{target_gene}.preds.tsv'
        if os.path.isfile(out_file):
            return None
        rna_seq_df, model = create_prediction_model(sample_file_name, target_gene, model, model_name)
        tissues = rna_seq_df.columns[1:-1]
        prediction_list = []
        std_dev_list = []
        # try:
        for tissue in tissues:
            res = predict_for_target_tissue(rna_seq_df, tissue, model)
            # prediction = res[0][0]
            # std_dev = res[1][0]
            # prediction_list.append(prediction)
            prediction_list.append(res[0])
            # std_dev_list.append(std_dev)

        preds = list(zip(tissues, prediction_list))
        # std_devs = list(zip(tissues, std_dev_list))
        fig, ax = plt.subplots()
        plt.bar(list(range(len(prediction_list))), prediction_list, width=0.8)
        ax.set_xticks(np.arange(len(prediction_list)))
        ax.set_xticklabels(list(tissues), rotation=90)
        ax.set_title(target_gene)
        # plt.show()
        plt.savefig(f'/cs/zbio/jrosensk/essentiality_predictions/{target_gene}.png')
        preds = sorted(preds, key=lambda tup: tup[1])
        top_preds = preds[0:5]

        with open(out_file, 'w') as f_out:
            to_write = target_gene + "\t" + "\t".join([a + ":" + str(b) for a, b in top_preds])
            f_out.write(to_write + "\n")
        # x = 0
        return preds, target_gene
    except Exception as e:
        print(e)
        print(target_gene)
        return None



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file',
                        default=None)
    parser.add_argument('--target_gene',
                        default='RPP25L')
    parser.add_argument('--model_name',
                        help="Options are 'linear', 'xg_boost', 'deep', 'ensemble', 'choose_best', 'GP'",
                        default='xg_boost')
    parser.add_argument('--expression_data_file',
                        default="gene_expression_tissue.descartes.csv")
    parser.add_argument('--tissue_name',
                        default=None)
    parser.add_argument('--log_output', help="A filename. default output is to std.out",
                        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict_all_tissues(args.expression_data_file, args.target_gene, args.model_name)
