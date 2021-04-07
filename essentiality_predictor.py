import argparse
import pandas as pd
import numpy as np
from Models import run_on_target
from configuration import gene_effect_file, gene_expression_file, ensemble_id_name_map_file


class EnsembleIDConverter:

    gene_column_name = "Name"
    gene_id_column_name = "geneID"
    gene_id_lookup = pd.read_csv(ensemble_id_name_map_file, sep='\t', names=[gene_column_name, gene_id_column_name], header=None)#"roadmap.TPM.tsv"
    gene_name_to_id_dict = dict(zip(gene_id_lookup.Name, gene_id_lookup.geneID))
    gene_id_to_name_dict = dict(zip(gene_id_lookup.geneID, gene_id_lookup.Name))

    def gene_id_to_name_lookup(self, gene_id):
        gene_id = gene_id.split(".")[0]
        if gene_id in self.gene_id_to_name_dict:
            return self.gene_id_to_name_dict[gene_id]
        else:
            return "not-found"


def get_sample_genes(rna_seq_df):
    sample_gene_id_column_name = "RowID"
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
                                                    model_name, None, num_folds=0, return_model=True,
                                                    genes_for_features=in_use_genes, use_knn=True)
    rna_seq = rna_seq.loc[rna_seq[EnsembleIDConverter().gene_column_name].isin(set(features))]
    rna_seq = rna_seq.sort_values(by=[EnsembleIDConverter().gene_column_name])
    return rna_seq, model


def predict_for_target_tissue(rna_seq_df, tissue_name, model):
    for_prediction = rna_seq_df[tissue_name]
    x = np.array([np.log2(np.array(for_prediction) + 1)])
    # x = model.enrich_with_knn(x)
    return model.predict_with_std(x)


def predict_all_tissues(sample_file_name, target_gene, model=None, model_name='GP'):
    rna_seq_df, model = create_prediction_model(sample_file_name, target_gene, model, model_name)
    tissues = rna_seq_df.columns[1:-1]
    prediction_list = []
    std_dev_list = []
    for tissue in tissues:
        res = predict_for_target_tissue(rna_seq_df, tissue, model)
        prediction = res[0][0]
        std_dev = res[1][0]
        prediction_list.append(prediction)
        std_dev_list.append(std_dev)

    preds = list(zip(tissues, prediction_list))
    std_devs = list(zip(tissues, std_dev_list))
    x = 0



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file',
                        default=None)
    parser.add_argument('--target_gene',
                        default='JUP')
    parser.add_argument('--model_name',
                        help="Options are 'linear', 'xg_boost', 'deep', 'ensemble', 'choose_best', 'GP'",
                        default='GP')
    parser.add_argument('--expression_data_file',
                        default="roadmap.tpms.tsv")
    parser.add_argument('--tissue_name',
                        default=None)
    parser.add_argument('--log_output', help="A filename. default output is to std.out",
                        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict_all_tissues(args.expression_data_file, args.target_gene)
