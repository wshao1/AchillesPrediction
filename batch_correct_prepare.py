from data_helper import get_intersecting_gene_ids_and_data
import pandas as pd
from essentiality_predictor import get_sample_genes


if __name__ == '__main__':
    achilles_scores, gene_expression, \
    train_test_df, cv_df = get_intersecting_gene_ids_and_data('Achilles_gene_effect.csv',
                                                              'CCLE_expression.csv',
                                                              cv_df_file="cross_validation_folds_ids.tsv",
                                                              train_test_df_file="train_test_split.tsv",
                                                              num_folds=1)
    rna_seq = pd.read_csv("gene_expression_tissue.descartes.csv")
    rna_seq = rna_seq.fillna(0)
    in_use_genes, rna_seq = get_sample_genes(rna_seq)

    intersecting_genes = set(gene_expression.columns).intersection(in_use_genes)

    rna_seq = rna_seq.loc[rna_seq["Name"].isin(intersecting_genes)]
    gene_expression = gene_expression[list(intersecting_genes)].T

    x = 0
