import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# from essentiality_predictor import EnsembleIDConverter
def calculate_tpm_roadmap():
    # data_df = pd.read_csv("/cs/zbio/jrosensk/road_map_rna_seq/read_count_matrix.ensemble.tsv", sep='\t')
    data_df = pd.read_csv("/cs/zbio/jrosensk/ccle_fastq/read_count_matrix.hg19.one_sample.tsv", sep="\t")
    counts_df = pd.read_csv("/cs/cbio/jon/projects/PyCharmProjects/AchillesPrediction/coding_lengths.hg19.tsv", sep="\t")
    ensembl_id_col_name = "Ensembl_gene_identifier"
    # ensemble_ids = []
    # for x in list(data_df["gene_name"]):
    #     splitted = x.split("(")
    #     if len(splitted) > 0:
    #         ensembl_id = splitted[1].strip()[:-1]
    #         ensemble_ids.append(ensembl_id)
    #     else:
    #         assert(False)
    # data_df[ensembl_id_col_name] = ensemble_ids
    ensemble_ids = []
    for x in list(data_df["gene_name"]):
        splitted = x.split(".")
        if len(splitted) > 0:
            ensembl_id = splitted[0].strip()
            ensemble_ids.append(ensembl_id)
        else:
            assert(False)
    data_df[ensembl_id_col_name] = ensemble_ids
    data_df.set_index(ensembl_id_col_name)
    counts_df.set_index(ensembl_id_col_name)
    data_df = data_df.merge(counts_df, on=ensembl_id_col_name)
    # gene_names = data_df["Name"]#.gene_name
    gene_lengths = data_df.length.values
    sample_names = ["neurospher"]#data_df.columns[1:-2]
    counts_matrix = data_df[[data_df.columns[1]]].values#data_df[data_df.columns[1:-2]].values
    read_per_gene_data = (counts_matrix.T / gene_lengths).T
    sum_of_proportions = np.sum(read_per_gene_data, axis=0) * (1 / 1e6)
    tpm = read_per_gene_data / sum_of_proportions
    tpm_df = pd.DataFrame(tpm, columns=sample_names)
    tpm_df["gene_names"] = data_df[ensembl_id_col_name]
    tpm_df.to_csv("/cs/zbio/jrosensk/road_map_rna_seq_bed/neurosphere.hg19.tpm.tsv", sep="\t", index=False)#"/cs/zbio/jrosensk/road_map_rna_seq/tpm_matrix_roadmap.ensemble.tsv"
    x = 0

def calculate_tpm_ccle():
    # data_df = pd.read_csv("/cs/zbio/jrosensk/road_map_rna_seq/read_count_matrix.ensemble.tsv", sep='\t')
    data_df = pd.read_csv("CENTRAL_NERVOUS_SYSTEM_hg19.ccle_lung.tsv")
    counts_df = pd.read_csv("/cs/cbio/jon/projects/PyCharmProjects/AchillesPrediction/coding_lengths.hg19.tsv", sep="\t")
    ensembl_id_col_name = "Ensembl_gene_identifier"
    # ensemble_ids = []
    # for x in list(data_df["gene_name"]):
    #     splitted = x.split("(")
    #     if len(splitted) > 0:
    #         ensembl_id = splitted[1].strip()[:-1]
    #         ensemble_ids.append(ensembl_id)
    #     else:
    #         assert(False)
    # data_df[ensembl_id_col_name] = ensemble_ids
    ensemble_ids = []
    for x in list(data_df[ensembl_id_col_name]):
        splitted = x.split(".")
        if len(splitted) > 0:
            ensembl_id = splitted[0].strip()
            ensemble_ids.append(ensembl_id)
        else:
            assert(False)
    data_df[ensembl_id_col_name] = ensemble_ids
    data_df.set_index(ensembl_id_col_name)
    counts_df.set_index(ensembl_id_col_name)
    data_df = data_df.merge(counts_df, on=ensembl_id_col_name)
    # gene_names = data_df["Name"]#.gene_name
    gene_lengths = data_df.length.values
    sample_names = ["CENTRAL_NERVOUS_SYSTEM"]#data_df.columns[1:-2]
    counts_matrix = data_df[["read_count"]].values#data_df[data_df.columns[1:-2]].values
    read_per_gene_data = (counts_matrix.T / gene_lengths).T
    sum_of_proportions = np.sum(read_per_gene_data, axis=0) * (1 / 1e6)
    tpm = read_per_gene_data / sum_of_proportions
    tpm_df = pd.DataFrame(tpm, columns=sample_names)
    tpm_df["gene_names"] = data_df[ensembl_id_col_name]
    tpm_df.to_csv("/cs/zbio/jrosensk/ccle_fastq/CENTRAL_NERVOUS_SYSTEM_hg19.tpm.tsv", sep="\t", index=False)#"/cs/zbio/jrosensk/road_map_rna_seq/tpm_matrix_roadmap.ensemble.tsv"
    x = 0


if __name__ == '__main__':
    calculate_tpm_ccle()
    # one = "/cs/zbio/jrosensk/ccle_fastq/CENTRAL_NERVOUS_SYSTEM_hg19.tpm.tsv"
    one = "road_map_developmental.tpms.tsv"
    two = "/cs/zbio/jrosensk/road_map_rna_seq_bed/neurosphere.hg19.tpm.tsv"
    ccle = pd.read_csv(one, sep="\t")
    ccle["Ensembl_gene_identifier"] = ccle["Gene ID"]
    # converter = EnsembleIDConverter()
    # ccle[converter.gene_column_name] = ccle["Ensembl_gene_identifier"].apply(
    #     lambda x: converter.gene_id_to_name_lookup(x))
    road_map = pd.read_csv(two, sep="\t")
    road_map["Ensembl_gene_identifier"] = road_map["gene_names"]
        # .apply(
        #  lambda x: x.split("(")[1][:-1])

    res = ccle.merge(road_map, on="Ensembl_gene_identifier")
    res = res.dropna()
    ccle_col = res["neurospher"]

    # others = res[res.columns[3:10]]
    for col in res.columns[2:-4]:
        # corr1, p_val1 = pearsonr(res[res.columns[3]], res[res.columns[5]])
        # corr3, p_val3 = pearsonr(res[res.columns[3]], res[res.columns[4]])
        corr2, p_val2 = pearsonr(ccle_col, res[col])
        x = 0
    # calculate_tpm_roadmap()