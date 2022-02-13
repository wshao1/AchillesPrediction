from FeatureSelection import get_features
from data_compare import read_gtex_file
from data_helper import get_intersecting_gene_ids_and_data
from sklearn.manifold import TSNE
from essentiality_predictor import get_sample_genes
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    achilles_scores, gene_expression, \
    train_test_df, cv_df = get_intersecting_gene_ids_and_data('Achilles_gene_effect.csv',
                                                              'CCLE_expression.csv',
                                                              cv_df_file="cross_validation_folds_ids.tsv",
                                                              train_test_df_file="train_test_split.tsv",
                                                              num_folds=1)
    # target_gene_name = "NPHS1"
    # y = achilles_scores[target_gene_name]
    # expression_feature_indices = get_features(y, gene_expression, 40)
    # feature_names = gene_expression.columns[expression_feature_indices]
    # features = list(set(list(feature_names) + [target_gene_name]))
    # features = list(np.var(gene_expression[gene_expression.columns[1:]]).sort_values(ascending=False).iloc[0:500].index)
    features_rpp25l = ['CRIP1', 'TNK1', 'CD164L2', 'RIBC2', 'SLC39A4', 'IFFO1', 'GLIPR2', 'ACOT4', 'FAM83H', 'RPP25',
                       'PRKCZ', 'RPP25L', 'LIPH', 'SPINT1', 'NAPRT', 'C11orf52', 'LSR', 'IQANK1', 'GRIN1', 'HPDL',
                       'CEACAM6', 'GSTO2', 'LLGL2', 'H2AJ', 'HENMT1', 'KRTCAP3', 'PFKFB2', 'CKMT1B', 'DNAJA4',
                       'SULT2B1', 'TJP3', 'TMEM30B', 'ADAP1', 'CLDN7', 'KLHL35', 'MARVELD2', 'CAPN8', 'TMC5', 'RFTN1',
                       'TMEM52', 'GRIN2D']
    features_vrk1 = ['SPPL2B', 'EDC4', 'SRCAP', 'UBA5', 'SNRNP70', 'RPL24', 'NKTR', 'CREB1', 'PEDS1-UBE2V1', 'INTS11',
                     'DGCR8', 'SAFB2', 'CCDC142', 'TSSK4', 'LENG8', 'TMEM262', 'LY6G5B', 'RBM14-RBM4', 'PIGY', 'EBLN2',
                     'AC022414.1', 'TBCE', 'SMIM41', 'AC008397.1', 'GCSAML-AS1', 'CCDC39', 'EEF1AKMT4-ECE2',
                     'AP000812.4', 'UPK3BL2', 'AC093512.2', 'ARHGAP11B', 'AC004593.2', 'AC090517.4', 'AL160269.1',
                     'ABCF2-H2BE1', 'POLR2J3', 'H2BE1', 'AL445238.1', 'GET1-SH3BGR', 'AC113348.1']
    features_chmp1a = ['EPN1', 'RRP15', 'SRCAP', 'SNRNP70', 'FBXL15', 'COX7A2', 'RPL24', 'CREB1', 'DDX54',
                       'PEDS1-UBE2V1', 'SAFB2', 'HSPBP1', 'RPS27', 'NELFB', 'RABL6', 'ZNF628', 'HYPK', 'RBM14-RBM4',
                       'PIGY', 'HSPE1-MOB4', 'AC022414.1', 'TBCE', 'SMIM41', 'AC008397.1', 'GCSAML-AS1', 'CCDC39',
                       'EEF1AKMT4-ECE2', 'AP000812.4', 'UPK3BL2', 'AC093512.2', 'ARHGAP11B', 'AC004593.2', 'AC090517.4',
                       'AL160269.1', 'ABCF2-H2BE1', 'POLR2J3', 'H2BE1', 'AL445238.1', 'GET1-SH3BGR', 'AC113348.1']

    blacklisted = set(['RPL24', 'TMEM262', 'PIGY', 'NKTR', 'SAFB2', 'PFKFB2', 'LIPH', 'ZNF628', 'TJP3', 'CRIP1', 'EBLN2', 'MARVELD2', 'GRIN2D', 'CREB1', 'RABL6', 'TMEM30B', 'IFFO1', 'GRIN1', 'GLIPR2', 'TSSK4', 'RBM14-RBM4', 'GSTO2', 'CAPN8', 'SRCAP', 'DGCR8', 'PRKCZ', 'KLHL35', 'CD164L2', 'RFTN1', 'ADAP1', 'TMC5', 'CKMT1B', 'CEACAM6', 'RIBC2', 'RPS27', 'CCDC39', 'FBXL15', 'DNAJA4', 'CCDC142', 'LLGL2'])
    features = list(set(features_rpp25l + features_vrk1 + features_chmp1a) - blacklisted)
    rna_seq = pd.read_csv("gene_expression_tissue.descartes.csv")
    rna_seq = rna_seq.fillna(0)
    in_use_genes, rna_seq = get_sample_genes(rna_seq)
    # features = list(
    #     np.var(rna_seq.set_index("Name").T).sort_values(ascending=False).iloc[0:1000].index)
    in_use_genes = set(features).intersection(set(in_use_genes))

    gene_column_name = "Name"
    rna_seq = rna_seq.loc[rna_seq[gene_column_name].isin(in_use_genes)]
    to_drop_ids = []

    seen_names = set()
    names = list(rna_seq["Name"])
    ids = list(rna_seq["RowID"])
    for cur_name, row_id in zip(names, ids):
        if cur_name in seen_names:
            to_drop_ids.append(row_id)
        else:
            seen_names.add(cur_name)
    rna_seq = rna_seq.loc[~rna_seq["RowID"].isin(set(to_drop_ids))]
    rna_seq = rna_seq[rna_seq.columns[1:]]


    descartes_expression = rna_seq.set_index("Name").T

    # gtex_info = pd.read_csv("/cs/staff/tommy/Work/GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    # tissue_col = "SMTS"
    # sample_id_col_name = "SAMPID"
    # tissue_sample_dict = {}
    # for index, row in gtex_info.iterrows():
    #     cur_sample_id = row[sample_id_col_name]
    #     cur_tissue = row[tissue_col]
    #     if cur_tissue not in tissue_sample_dict:
    #         tissue_sample_dict[cur_tissue] = [cur_sample_id]
    #     else:
    #         cur_samples = tissue_sample_dict[cur_tissue]
    #         tissue_sample_dict[cur_tissue] = cur_samples + [cur_sample_id]
    # del gtex_info
    # sample_ids = tissue_sample_dict["Kidney"]
    # gtex = read_gtex_file(in_use_genes, sample_ids + ["Name"])
    # to_drop_ids = []
    # seen_names = set()
    # names = list(gtex["Description"])
    # ids = list(gtex["Name"])
    # for cur_name, row_id in zip(names, ids):
    #     if cur_name in seen_names:
    #         to_drop_ids.append(row_id)
    #     else:
    #         seen_names.add(cur_name)
    # gtex = gtex.loc[~gtex["Name"].isin(set(to_drop_ids))]
    # gtex = gtex.set_index("Description").T
    # in_use_genes = set(in_use_genes).intersection(set(gtex.columns))
    sorted_genes = sorted(list(in_use_genes))
    # print("number of genes is " + str(len(sorted_genes)))

    gene_expression = np.array(gene_expression[sorted_genes])
    descartes_expression = descartes_expression[sorted_genes]
    # gtex = gtex[sorted_genes]
    # assert(gtex.shape[1] == descartes_expression.shape[1])

    m = 0.235969463216031
    b = 13.05616897327613
    descartes_expression_c = (descartes_expression ) / m
    descartes_expression = np.log2(np.array(descartes_expression_c) + 1)
    # gtex = np.array(gtex)[1:, :]
    # idx = np.random.randint(gtex.shape[0], size=300)
    # gtex = gtex[idx,:].astype(float)
    # gtex = np.log2(gtex + 1)

    # total_data = np.concatenate((gene_expression, descartes_expression, gtex), axis=0)
    total_data = np.concatenate((gene_expression, descartes_expression), axis=0)

    tsne = TSNE(n_components=2, random_state=0)
    total_data = tsne.fit_transform(total_data)

    gene_expression_rows = gene_expression.shape[0]
    descartes_end = gene_expression_rows + descartes_expression.shape[0]

    plt.scatter(total_data[0:gene_expression_rows, 0], total_data[0:gene_expression_rows, 1], c='r',
                label="achilles")
    plt.scatter(total_data[gene_expression_rows:descartes_end, 0], total_data[gene_expression_rows:descartes_end, 1],
                c='b',
                label="descartes")
    # plt.scatter(total_data[descartes_end:, 0], total_data[descartes_end:, 1],
    #             c='orange',
    #             label="gtex")
    plt.legend()
    plt.show()

    x = 0
