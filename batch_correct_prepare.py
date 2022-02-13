import pandas as pd
import numpy as np


if __name__ == '__main__':
    rna_seq = pd.read_csv("gtex_read_counts.csv")
    # gene_expression = pd.read_csv("CCLE_RNAseq_reads.csv")
    ccle_genes = pd.read_csv('CCLE_RNAseq_reads.csv', nrows=1)
    ccle_genes = list(ccle_genes.columns)
    ccle_ensemble_ids = []
    for entry in ccle_genes:
        tokens = entry.split("(")
        if len(tokens) > 1:
            entry = tokens[1][:-1]
            ccle_ensemble_ids.append(entry)
        else:
            ccle_ensemble_ids.append(entry)
    # rna_seq = pd.read_csv("gtex_read_counts.csv")
    pc_path = '57epigenomes.N.pc.gz'
    roadmap_dir = '/cs/cbio/hadar/data/roadmap/'
    data_df = pd.read_csv(roadmap_dir + pc_path, sep='\t', index_col=0, skiprows=1, header=None).iloc[:,:-1]
    gtex_ensemble = list(data_df.index)
    interescting_ids = set(gtex_ensemble).intersection(set(ccle_ensemble_ids))

    in_use_ccle_columns = []
    for entry in ccle_genes:
        tokens = entry.split("(")
        if len(tokens) > 1:
            to_check = tokens[1][:-1]
        else:
            to_check = entry
        if to_check in interescting_ids:
            in_use_ccle_columns.append(entry)
    ccle_expression = pd.read_csv('CCLE_RNAseq_reads.csv', usecols=in_use_ccle_columns)

    data_df = data_df.loc[data_df.index.isin(set(interescting_ids))]
    new_cols = []
    ensemble_id_name_map = {}
    for entry in ccle_expression.columns:
        tokens = entry.split("(")
        if len(tokens) > 1:
            ensemb_id = tokens[1][:-1]
            gene_name = tokens[0].strip()
        else:
            if entry.startswith("ENS"):
                ensemb_id = entry.strip()
            else:
                ensemb_id = None
            gene_name = entry.strip()
        new_cols.append(gene_name)
        if ensemb_id is not None:
            ensemble_id_name_map[ensemb_id] = gene_name
    ccle_expression.columns = new_cols
    new_gtex_gene_names = []
    for entry in data_df.index:
        new_gtex_gene_names.append(ensemble_id_name_map[entry])

    data_df["gene_name"] = new_gtex_gene_names
    data_df = data_df.reset_index().drop([0], axis=1)
    data_df = data_df.set_index('gene_name').T
    sorted_genes = sorted(list(new_gtex_gene_names))
    data_df = data_df[sorted_genes]
    ccle_expression = ccle_expression[sorted_genes]
    data_df = data_df.T
    ccle_expression = ccle_expression.T

    res = pd.concat([ccle_expression, data_df.et_index(ccle_expression.index)], axis=1)
    batch_vector = np.ones(res.shape[1])
    batch_vector[ccle_expression.shape[1]:] = 2
    batch_df = pd.DataFrame(batch_vector)
    res.to_csv("for_batch_correction.csv", index=False)
    batch_df.to_csv("batch_vector.csv", index=False)
    x = 0
    # rna_seq = rna_seq.fillna(0)

    # seen_names = set()
    # to_drop_ids = []
    # names = list(rna_seq["Name"])
    # ids = list(rna_seq["RowID"])
    # for cur_name, row_id in zip(names, ids):
    #     if cur_name in seen_names:
    #         to_drop_ids.append(row_id)
    #     else:
    #         seen_names.add(cur_name)
    # rna_seq = rna_seq.loc[~rna_seq["RowID"].isin(set(to_drop_ids))]
    # rna_seq = rna_seq[rna_seq.columns[1:]].set_index('Unnamed: 0').T
    #
    # sorted_genes = sorted(list(intersecting_genes))
    # gene_expression = gene_expression[list(sorted_genes)].T
    # gene_expression = np.exp2(gene_expression) - 1
    # rna_seq = rna_seq[list(sorted_genes)].T
    # res = pd.concat([gene_expression, rna_seq.set_index(gene_expression.index)], axis=1)
    #
    # batch_vector = np.ones(res.shape[1])
    # batch_vector[gene_expression.shape[1]:] = 2
    # batch_df = pd.DataFrame(batch_vector)
    # res.to_csv("for_batch_correction.csv", index=False)
    # batch_df.to_csv("batch_vector.csv", index=False)
    # x = 0
