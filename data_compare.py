import pandas as pd
from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
from data_helper import get_intersecting_gene_ids_and_data, clean_gene_names
import math
import heapq
import gzip as gz
import numpy as np
from numpy.polynomial.polynomial import polyfit


from essentiality_predictor import get_sample_genes, EnsembleIDConverter


def calc_distance(x, y, p=2):
    sum = 0
    for i, j in zip(x, y):
        sum += (i - j) ** p
    return sum # ** (1/p)


def calc_avg_distance(X, Y, p=2):

    if len(X.shape) == 1:
        X = [X]
    if len(Y.shape) == 1:
        Y = [Y]
    n = 0
    distance = 0
    for x in X:
        for y in Y:
            distance += calc_distance(x, y, p)
            n += 1
    if n == 0:
        return math.nan
    return distance / n


def calc_avg_distance_nearest_k(X, Y, k=3, p=2):
    if len(X.shape) == 1:
        X = [X]
    if len(Y.shape) == 1:
        Y = [Y]
    n = 0
    distance = 0
    for x in X:
        neighbors = []
        for y in Y:
            cur_distance = calc_distance(x, y, p)
            if len(neighbors) < k:
                heapq.heappush(neighbors, (-cur_distance, y))
            else:
                heapq.heappushpop(neighbors, (-cur_distance, y))
        n += 1
        distance += np.sum([-dist for dist, y in neighbors]) / k
    if n == 0:
        return math.nan
    return distance / n


def read_gtex_file(features, sample_ids):
    features = set(features)
    is_first = True
    rows = []
    with gz.open("/cs/staff/tommy/Work/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz", 'r') as f:
        for line in f:
            line = line.decode().rstrip()
            tokens = line.split("\t")
            if len(tokens) > 5:
                if is_first:
                    is_first = False
                    gtex_col_set = set(tokens)
                    columns = sample_ids + ["Description"]
                    in_use_columns = gtex_col_set.intersection(set(columns))
                    in_use_col_indices = []
                    col_names = []

                    for idx, col_name in enumerate(tokens):
                        if col_name in in_use_columns:
                            in_use_col_indices.append(idx)
                            col_names.append(col_name)
                        if col_name == "Description":
                            description_index = idx
                    in_use_col_indices = set(in_use_col_indices)
                    continue
                cur_row = []
                if tokens[description_index] in features:
                    for idx, el in enumerate(tokens):
                        if idx in in_use_col_indices:
                            if idx > 1:
                                el = float(el)
                            cur_row.append(el)
                    rows.append(cur_row)
    df = pd.DataFrame(rows, columns=col_names)
    return df


def qq_plots(x, y):
    assert(x.shape[0] > y.shape[0])
    num_samples = y.shape[0]
    x = np.sort(x)
    y = np.sort(y)
    x_percentiles = np.zeros(num_samples)
    for i in range(num_samples):
        cur_percent = float(i + 1) / float(num_samples)
        x_rank_fraction = cur_percent * (x.shape[0] - 1)
        lower_index = int(np.floor(x_rank_fraction))
        upper_index = lower_index + 1
        if upper_index < x.shape[0]:
            upper_frac = np.ceil(x_rank_fraction) - x_rank_fraction
            lower_frac = 1.0 - upper_frac
            cur_percentil_val = (x[lower_index] * lower_frac) + (x[upper_index] * upper_frac)
        else:
            cur_percentil_val = x[lower_index]
        x_percentiles[i] = cur_percentil_val

    fig, ax = plt.subplots()
    plt.scatter(x_percentiles, y, c='r')
    ax.set_xlabel('DepMap')
    ax.set_ylabel('Descartes')
    ax.set_title('Percentiles DepMap vs Descartes: {} points'.format(str(num_samples)))

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    x = 0



def get_xy_points_for_tissue(dep_map_expression, descartes_expression, depmap_tissue_names, descartes_tissue_names):
    gene_expression = dep_map_expression.copy()
    descartes_data = descartes_expression.copy()
    descartes_data = descartes_data[["gene_id"] + descartes_tissue_names]
    gene_expression = gene_expression[gene_expression["tissue_types"].isin(set(depmap_tissue_names))]

    del gene_expression['tissue_types']

    gene_expression = gene_expression.set_index("Unnamed: 0").T
    gene_expression = gene_expression.reset_index()
    gene_expression.columns = ["gene_id"] + list(gene_expression.columns[1:])
    gene_expression["gene_id"] = gene_expression["gene_id"].apply(
        lambda x: x.split("(")[0].strip())
    converter = EnsembleIDConverter()
    gene_expression["gene_id"] = gene_expression["gene_id"].apply(
        lambda x: converter.gene_name_to_id_lookup(x))
    gene_expression = gene_expression[gene_expression["gene_id"] != "not-found"]
    ccle_genes = set(gene_expression["gene_id"])

    descartes_genes = set(descartes_data["gene_id"])

    descartes_data = descartes_data.loc[descartes_data["gene_id"].isin(set(ccle_genes))]
    gene_expression = gene_expression.loc[gene_expression["gene_id"].isin(set(descartes_genes))]
    assert (len(list(gene_expression["gene_id"])) == len(list(descartes_data["gene_id"])))
    assert (set(gene_expression["gene_id"]) == set(descartes_data["gene_id"]))
    descartes_data = descartes_data.sort_values(by="gene_id")
    gene_expression = gene_expression.sort_values(by="gene_id")

    # qq_plots(np.exp2(list(gene_expression.iloc[0, 1:])) - 1, np.array(descartes_data.iloc[0, 1:]))

    gene_ids = list(gene_expression["gene_id"])
    blacklisted_genes = []
    ccle_means = np.zeros(gene_expression.shape[0])
    descartes_means = np.zeros(gene_expression.shape[0])
    for i in range(gene_expression.shape[0]):
        depmap_values = np.exp2(list(gene_expression.iloc[i, 1:])) - 1
        ccle_mean_cur = np.sum(depmap_values) / (gene_expression.shape[1] - 1)
        descartes_mean_cur = np.sum(descartes_data.iloc[i, 1:]) / (descartes_data.shape[1] - 1)
        cond1 = ((descartes_mean_cur / ccle_mean_cur) < 2) and ((ccle_mean_cur / descartes_mean_cur) < 50)
        cond2 = descartes_mean_cur < 250 and ccle_mean_cur < 1000
        if cond1 and cond2:
            ccle_means[i] = ccle_mean_cur
            descartes_means[i] = descartes_mean_cur
        else:
            blacklisted_genes.append(gene_ids[i])
            ccle_means[i] = -1
            descartes_means[i] = -1

    ccle_means = np.array([x for x in ccle_means if x > -1])
    descartes_means = np.array([x for x in descartes_means if x > -1])
    return ccle_means, descartes_means, blacklisted_genes




if __name__ == '__main__':
    # select features
    # make two dfs only with selected features
    # group by tisues
    # for each pair in two data sets calculate distance

    features_rpp25l = ['CRIP1', 'TNK1', 'CD164L2', 'RIBC2', 'SLC39A4', 'IFFO1', 'GLIPR2', 'ACOT4', 'FAM83H', 'RPP25', 'PRKCZ', 'RPP25L', 'LIPH', 'SPINT1', 'NAPRT', 'C11orf52', 'LSR', 'IQANK1', 'GRIN1', 'HPDL', 'CEACAM6', 'GSTO2', 'LLGL2', 'H2AJ', 'HENMT1', 'KRTCAP3', 'PFKFB2', 'CKMT1B', 'DNAJA4', 'SULT2B1', 'TJP3', 'TMEM30B', 'ADAP1', 'CLDN7', 'KLHL35', 'MARVELD2', 'CAPN8', 'TMC5', 'RFTN1', 'TMEM52', 'GRIN2D']
    features_vrk1 = ['SPPL2B', 'EDC4', 'SRCAP', 'UBA5', 'SNRNP70', 'RPL24', 'NKTR', 'CREB1', 'PEDS1-UBE2V1', 'INTS11', 'DGCR8', 'SAFB2', 'CCDC142', 'TSSK4', 'LENG8', 'TMEM262', 'LY6G5B', 'RBM14-RBM4', 'PIGY', 'EBLN2', 'AC022414.1', 'TBCE', 'SMIM41', 'AC008397.1', 'GCSAML-AS1', 'CCDC39', 'EEF1AKMT4-ECE2', 'AP000812.4', 'UPK3BL2', 'AC093512.2', 'ARHGAP11B', 'AC004593.2', 'AC090517.4', 'AL160269.1', 'ABCF2-H2BE1', 'POLR2J3', 'H2BE1', 'AL445238.1', 'GET1-SH3BGR', 'AC113348.1']
    features_chmp1a = ['EPN1', 'RRP15', 'SRCAP', 'SNRNP70', 'FBXL15', 'COX7A2', 'RPL24', 'CREB1', 'DDX54', 'PEDS1-UBE2V1', 'SAFB2', 'HSPBP1', 'RPS27', 'NELFB', 'RABL6', 'ZNF628', 'HYPK', 'RBM14-RBM4', 'PIGY', 'HSPE1-MOB4', 'AC022414.1', 'TBCE', 'SMIM41', 'AC008397.1', 'GCSAML-AS1', 'CCDC39', 'EEF1AKMT4-ECE2', 'AP000812.4', 'UPK3BL2', 'AC093512.2', 'ARHGAP11B', 'AC004593.2', 'AC090517.4', 'AL160269.1', 'ABCF2-H2BE1', 'POLR2J3', 'H2BE1', 'AL445238.1', 'GET1-SH3BGR', 'AC113348.1']
    features_tp53 = ['ACTA2', 'ANXA5', 'BAX', 'CALD1', 'CCDC88C', 'CCNG1', 'CD63', 'CDKN1A', 'CNKSR1', 'CRTAP', 'DDB2', 'DEF6', 'DGKE', 'DRAM1', 'ECM1', 'EDA2R', 'GALNT6', 'GRK2', 'HTRA1', 'IGSF9', 'IKBIP', 'MCM2', 'MDM2', 'MRAS', 'OR4F15', 'OSTM1', 'PCDHGB7', 'PDLIM2', 'PHLDA3', 'PMP2', 'PMP22', 'PROB1', 'PTCHD4', 'PTK2B', 'RHOQ', 'RPS27L', 'RRM2B', 'SASS6', 'SERPINE2', 'SESN1', 'SHISA4', 'SPATA18', 'SUGCT', 'TIGAR', 'TMEM216', 'TNFRSF10B', 'TP53INP1', 'TSEN54', 'VIM', 'ZMAT3']
    features_nhlrc2 = ['ABL2', 'AGR2', 'ARHGAP27', 'BCAP29', 'C10orf90', 'CALD1', 'CALU', 'CCDC187', 'CCER2', 'CHMP4C', 'CNN3', 'CNRIP1', 'CRTAP', 'CYB5R3', 'CYP20A1', 'DACT1', 'DCT', 'DGKI', 'DLC1', 'DOCK7', 'FAAH', 'FAM180A', 'FMNL3', 'FN1', 'FRMD4A', 'GDNF', 'IL13RA2', 'IL24', 'ITGB3', 'JAG2', 'KIAA1549L', 'KIAA1755', 'KLF17', 'MAP4', 'MFSD12', 'MICAL3', 'MMP8', 'MPP4', 'NRG4', 'OPN1SW', 'OSTM1', 'PHACTR1', 'PMP22', 'PTPN6', 'RHPN1', 'RNF183', 'SEMA4D', 'SH2B3', 'SLC15A2', 'SLC39A14', 'SMTN', 'SOGA3', 'SPARC', 'SRPX', 'SULT1A1', 'SYDE1', 'SYTL1', 'TIMP2', 'TRAK2', 'TUBB2A', 'TYR', 'VIM', 'ZNF697']
    features_mitf = ['ABCB5', 'ACP5', 'ALX1', 'ARMC9', 'BBS5', 'BCAN', 'BIRC7', 'C10orf90', 'CAPN3', 'CDH19', 'CHL1', 'CPB2', 'CPN1', 'ENTHD1', 'EXTL1', 'FABP7', 'FAM180B', 'FAXC', 'GAPDHS', 'GJB1', 'HTN1', 'IGSF11', 'KCNH1', 'KLHL38', 'KRTAP19-1', 'LGI3', 'LRRK2', 'MDGA2', 'MITF', 'MLANA', 'MMP8', 'MRGPRX4', 'OR5J2', 'OR9G1', 'PLA1A', 'PLOD3', 'PLP1', 'PMEL', 'PRDM7', 'PRY', 'PTCRA', 'RAB38', 'RGR', 'RLBP1', 'ROPN1', 'ROPN1B', 'RXRG', 'S100B', 'SFTPC', 'SGCD', 'SLC24A5', 'SLC35B2', 'SLC45A2', 'SLC6A15', 'SMYD1', 'SNCA', 'SOX10', 'SPATS1', 'TRIM51', 'TRIM63', 'TRPM1', 'TSPAN10', 'TYR']
    features_DNAJC19 = ['ABI2', 'ACY3', 'ALDH3A1', 'ATP1B2', 'BATF', 'BTBD17', 'C16orf54', 'CABP7', 'CCRL2', 'CLIP3', 'COIL', 'CSRNP2', 'CTSS', 'DCLK3', 'DHX40', 'DLX5', 'DNAJC15', 'DRAXIN', 'EIF4E1B', 'EML2', 'EPSTI1', 'ERC2', 'FAM184A', 'FRS2', 'GPATCH2', 'GPR162', 'GPR22', 'GPR26', 'GRIK5', 'HUNK', 'KCNH7', 'MBTD1', 'MEX3A', 'MSI1', 'MTA3', 'MYH10', 'MYL6B', 'NOVA1', 'OGDHL', 'PDE1A', 'PPP1R17', 'PREPL', 'PYCARD', 'QPCT', 'RIMKLB', 'RIMS1', 'RIPK3', 'RPAIN', 'RTP5', 'SALL2', 'SCN8A', 'SIAH3', 'SIDT1', 'SLC37A2', 'STXBP4', 'SVBP', 'THBS3', 'UCHL1', 'WASF1', 'WNK3', 'ZBP1', 'ZFHX2', 'ZMYM4', 'ZNF140', 'ZNF529', 'ZNF627', 'ZNF713', 'ZNF84']
    features = list(set(features_rpp25l + features_vrk1 + features_chmp1a + features_tp53 + features_nhlrc2 + features_mitf + features_DNAJC19))
    achilles_scores, gene_expression, \
    train_test_df, cv_df = get_intersecting_gene_ids_and_data('Achilles_gene_effect.csv',
                                                              'CCLE_expression.csv',
                                                              cv_df_file="cross_validation_folds_ids.tsv",
                                                              train_test_df_file="train_test_split.tsv",
                                                              num_folds=1)
    del achilles_scores, train_test_df, cv_df
    gene_expression = gene_expression[features + ["Unnamed: 0"]]

    cell_ids = list(gene_expression["Unnamed: 0"])
    # achilles_expression = gene_expression[features]
    sample_info = pd.read_csv("sample_info.csv")
    tissue_types = []
    for cell_id in cell_ids:
        cur_tissue = list(sample_info[['DepMap_ID', 'sample_collection_site']][
                              sample_info.DepMap_ID == cell_id].sample_collection_site)[0]
        tissue_types.append(cur_tissue)

    gene_expression['tissue_types'] = tissue_types
    tissue_set_achilles = set(tissue_types)



    # gene_expression = pd.read_csv('CCLE_expression.csv')
    # as_mat = gene_expression.values
    # count_zero = 0
    # for i in range(gene_expression.shape[1] - 1):
    #     cur_sum = np.sum(as_mat[:, i + 1])
    #     if cur_sum == 0:
    #         count_zero += 1
    #         y = 0



    # expression_id_col_name = 'Unnamed: 0'
    # gene_expression = clean_gene_names(gene_expression, expression_id_col_name)
    # features = set(gene_expression.columns)

    ############## GTEX

    # gtex_info = pd.read_csv("/cs/staff/tommy/Work/GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    # tissue_col = "SMTS"
    # sample_id_col_name = "SAMPID"
    #
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
    #
    #
    # # count = 0
    # # with gz.open("/cs/staff/tommy/Work/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz", 'r') as f:
    # #     for line in f:
    # #         line = line.decode().rstrip()
    # #         if count == 2:
    # #             gtex_columns = line.split("\t")
    # #             break
    # #         elif count < 2:
    # #             count += 1
    # #         else:
    # #             break
    # # gtex_col_set = set(gtex_columns)
    # x = 0
    # tissue_distances = {}
    # for gtex_tissue, sample_ids in tissue_sample_dict.items():
    #     gtex = read_gtex_file(features, sample_ids)
    #     # columns = sample_ids + ["Description"]
    #     # columns_set = list(gtex_col_set.intersection(set(columns)))
    #     # print("expected {} samples but only have {}".format(str(len(columns)), str(len(columns_set))))
    #     # gtex = pd.read_csv("/cs/staff/tommy/Work/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz",
    #     #                    sep="\t", skiprows=2, usecols=columns_set)
    #     # gtex = gtex.loc[gtex["Description"].isin(set(features))]
    #     #
    #     # print("expected {} feature and have {}".format(str(len(features)), str(gtex.shape[0])))
    #     gtex = gtex.set_index("Description").T
    #     features = set(features).intersection(set(gtex.columns))
    #     sorted_features = sorted(list(features))
    #     gtex = gtex[sorted_features]
    #     cur_gtex_tissue = np.log2(np.array(gtex) + 1)
    #     for achilles_tissue in tissue_set_achilles:
    #         achilles_cur_tissue_expression = gene_expression[gene_expression['tissue_types'] == achilles_tissue]
    #         achilles_cur_tissue_expression = achilles_cur_tissue_expression[sorted_features]
    #         achilles_cur_tissue_expression = np.array(achilles_cur_tissue_expression)
    #         avg_distance = calc_avg_distance_nearest_k(cur_gtex_tissue, achilles_cur_tissue_expression)
    #         key = (gtex_tissue, achilles_tissue)
    #         tissue_distances[key] = avg_distance

    ############# Descartes
    descartes_data = pd.read_csv("gene_expression_tissue.descartes.csv")
    descartes_data = descartes_data.dropna()
    descartes_data.columns = ["gene_id"] + list(descartes_data.columns[1:])

    tissue_pairs = [(['large_intestine', 'small_intestine'], ['Intestine']), (['liver'], ['Liver']),
                    (["central_nervous_system"], ["Cerebellum", "Cerebrum"]), (["stomach"], ["Stomach"]),
                    (["lung"], ["Lung"])]

    # in_use_genes, descartes_data = get_sample_genes(descartes_data)
    descartes_data["gene_id"] = descartes_data["gene_id"].apply(
        lambda x: x.split(".")[0])
    assert(len(descartes_data["gene_id"]) == len(set(descartes_data["gene_id"])))

    ccle_means = []
    descartes_means = []
    blacklisted_genes = set()
    for ccle_tissues, descartes_tissues in tissue_pairs:
        ccle_means_cur, descartes_means_cur, blacklisted_genes_cur = get_xy_points_for_tissue(gene_expression, descartes_data, ccle_tissues, descartes_tissues)
        for a, b in zip(ccle_means_cur, descartes_means_cur):
            ccle_means.append(a)
            descartes_means.append(b)
        blacklisted_genes.update(blacklisted_genes_cur)

    converter = EnsembleIDConverter()
    blacklisted_names = [converter.gene_id_to_name_lookup(x) for x in blacklisted_genes]
    # depmap_values = np.exp2(list(gene_expression.iloc[30, 1:])) - 1
    # ccle_mean_rpp25 = np.array(np.sum(depmap_values) / (gene_expression.shape[1] - 1))
    # descartes_mean_rpp25 = np.array(np.sum(descartes_data.iloc[30, 3:5]) / (descartes_data.shape[1] - 1))

    x = np.array(ccle_means) # + list(ccle_mean_rpp25.flatten())
    y = np.array(descartes_means)
    b, m = polyfit(x, y, 1)
    print("b is: " + str(b))
    print("m is: " + str(m))
    descartes_transformed = (descartes_means) / m
    x2 = np.array(ccle_means)
    y2 = np.array(descartes_transformed)
    b2, m2 = polyfit(x2, y2, 1)

    fig, ax = plt.subplots()
    plt.scatter(ccle_means, descartes_means, c='r')
    ax.set_xlabel('DepMap genes')
    ax.set_ylabel('Descartes genes')
    ax.set_title('RPP25L Modifier genes DepMap vs Descartes expression mean')


    plt.plot(x, b + m * x, '--')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    plt.scatter(ccle_means, descartes_transformed, c='r')
    ax2.set_xlabel('DepMap genes')
    ax2.set_ylabel('Descartes genes Transformed')
    ax2.set_title('RPP25L Modifier Descartes expression mean Transformed')
    plt.plot(x2, b2 + m2 * x2, '--')

    # Tweak spacing to prevent clipping of ylabel
    fig2.tight_layout()
    plt.show()
    x = 0

    # ccle_lung = pd.read_csv("/cs/zbio/jrosensk/ccle_fastq/CENTRAL_NERVOUS_SYSTEM_hg19.tpm.tsv", sep="\t")
    # ccle_lung.columns = list(ccle_lung.columns[:-1]) + ["gene_id"]
    # ccle_lung = ccle_lung.dropna()
    # ccle_lung = ccle_lung.groupby(['gene_id']).sum()
    # ccle_lung = ccle_lung.reset_index()
    # assert (len(ccle_lung["gene_id"]) == len(set(ccle_lung["gene_id"])))
    # ccle_genes = set(ccle_lung["gene_id"])
    # x=0

    gtex = pd.read_csv("/cs/staff/tommy/Work/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz", sep="\t", skiprows=[0,1], usecols=["Name", "GTEX-1117F-0226-SM-5GZZ7", "GTEX-1117F-0426-SM-5EGHI", "GTEX-1122O-1226-SM-5H113", "GTEX-11DXZ-0226-SM-5EGGZ"])
    gtex.columns = ["gene_id"] + list(gtex.columns[1:])
    gtex["gene_id"] = gtex["gene_id"].apply(
        lambda x: x.split(".")[0])
    gtex = gtex.dropna()
    gtex = gtex.groupby(['gene_id']).sum()

    gtex = gtex.reset_index()
    assert (len(gtex["gene_id"]) == len(set(gtex["gene_id"])))
    gtex_genes = set(gtex["gene_id"])
    x = 0

    # road_map_data = pd.read_csv("road_map_developmental.tpms.tsv", sep="\t")
    # road_map_data.columns = ["gene_id"] + list(road_map_data.columns[1:])
    # road_map_data = road_map_data.dropna()
    # for col in road_map_data.columns[2:]:
    #     x = sum(road_map_data[col])
    #     assert(abs(sum(road_map_data[col])-1e6) < 200000)
    # for col in descartes_data.columns[1:]:
    #     x = sum(descartes_data[col])
    #     assert(abs(sum(descartes_data[col])-1e6) < 10)
    # for col in gtex.columns[1:]:
    #     x = sum(gtex[col])
    #     assert(abs(sum(gtex[col])-1e6) < 20)

    gtex = gtex.loc[gtex["gene_id"].isin(set(descartes_genes))]
    gtex = gtex.loc[gtex["gene_id"].isin(set(ccle_genes))]
    descartes_data = descartes_data.loc[descartes_data["gene_id"].isin(set(gtex_genes))]
    descartes_data = descartes_data.loc[descartes_data["gene_id"].isin(set(ccle_genes))]
    # ccle_lung = ccle_lung.loc[ccle_lung["gene_id"].isin(set(gtex_genes))]
    # ccle_lung = ccle_lung.loc[ccle_lung["gene_id"].isin(set(descartes_genes))]
    assert(len(list(gtex["gene_id"])) == len(list(descartes_data["gene_id"])))
    assert (set(gtex["gene_id"]) == set(descartes_data["gene_id"]))
    # assert (len(list(ccle_lung["gene_id"])) == len(list(descartes_data["gene_id"])))
    # assert (set(ccle_lung["gene_id"]) == set(ccle_lung["gene_id"]))
    descartes_data = descartes_data.sort_values(by="gene_id")
    gtex = gtex.sort_values(by="gene_id")
    gene_expression = gene_expression.sort_values(by="gene_id")
    correlations = np.zeros([descartes_data.shape[1]-1, gtex.shape[1]-1])
    p_vals = np.zeros([descartes_data.shape[1] - 1, gtex.shape[1] - 1])
    # for i in range(descartes_data.shape[1]-1):
    #     for j in range(gtex.shape[1]-1):
    #         d_vals = list(descartes_data[descartes_data.columns[i+1]])
    #         gtex_vals = list(gtex[gtex.columns[j+1]])
    #         cur_pear, p_val = spearmanr(gtex_vals, d_vals)
    #         if abs(cur_pear) > 0.85:
    #             y = 0
    #         correlations[i,j] = cur_pear
    #         p_vals[i, j] = p_val

    descartes_ccle_p_vals = np.zeros([descartes_data.shape[1]-1, gene_expression.shape[1]-1])
    gtex_ccle_p_vals = np.zeros([gtex.shape[1] - 1, gene_expression.shape[1]-1])
    descartes_ccle_correlations = np.zeros([descartes_data.shape[1] - 1, gene_expression.shape[1]-1])
    gtex_ccle_correlations = np.zeros([gtex.shape[1] - 1, gene_expression.shape[1]-1])

    for i in range(descartes_data.shape[1]-1):
        for j in range(gene_expression.shape[1]-1):
            ccle_col = list(np.exp2(gene_expression[gene_expression.columns[j+1]]) - 1)
            d_vals = list(descartes_data[descartes_data.columns[i+1]])
            cur_pear, p_val = spearmanr(d_vals, ccle_col)
            if abs(cur_pear) > 0.85:
                y = 0
            descartes_ccle_p_vals[i, j] = p_val
            descartes_ccle_correlations[i, j] = cur_pear

    for i in range(gtex.shape[1]-1):
        for j in range(gene_expression.shape[1] - 1):
            ccle_col = list(np.exp2(gene_expression[gene_expression.columns[j+1]]) - 1)
            gtex_vals = list(gtex[gtex.columns[i+1]])
            cur_pear, p_val = spearmanr(gtex_vals, ccle_col)
            if abs(cur_pear) > 0.85:
                y = 0
            gtex_ccle_p_vals[i, j] = p_val
            gtex_ccle_correlations[i, j] = cur_pear

    x = 0
    # gene_column_name = "Name"
    # rna_seq = descartes_data.loc[descartes_data[gene_column_name].isin(set(features))]
    #
    # to_drop_ids = []
    # seen_names = set()
    # names = list(rna_seq["Name"])
    # ids = list(rna_seq["RowID"])
    # for cur_name, row_id in zip(names, ids):
    #     if cur_name in seen_names:
    #         to_drop_ids.append(row_id)
    #     else:
    #         seen_names.add(cur_name)
    # rna_seq = rna_seq.loc[~rna_seq["RowID"].isin(set(to_drop_ids))]
    # rna_seq = rna_seq[rna_seq.columns[1:]]
    #
    # gtex = rna_seq.set_index("Name").T
    #
    # # tissue_distances = {}
    # features = set(features).intersection(set(gtex.columns))
    # sorted_features = sorted(list(features))
    # gtex = gtex[sorted_features]
    # gene_expression = gene_expression[sorted_features]
    # logged_gtex = np.log2(np.array(gtex) + 1)
    # logged_ccle = np.array(gene_expression)
    # correlations = np.zeros([gene_expression.shape[0], gene_expression.shape[0]])
    # # for i in range(gene_expression.shape[0]):
    # #     for j in range(gtex.shape[0]):
    # #         gene_expression_vec = logged_ccle[i, :]
    # #         other_vec = logged_gtex[j, :]
    # #         cur_pear, p_val = pearsonr(gene_expression_vec, other_vec)
    # #         if abs(cur_pear) > 0.85:
    # #             y = 0
    # #         correlations[i,j] = cur_pear
    #
    # for i in range(gene_expression.shape[0]):
    #     for j in range(gene_expression.shape[0]):
    #         gene_expression_vec = logged_ccle[i, :]
    #         other_vec = logged_ccle[j, :]
    #         cur_pear, p_val = pearsonr(gene_expression_vec, other_vec)
    #         if abs(cur_pear) > 0.85:
    #             y = 0
    #         correlations[i,j] = cur_pear
    #
    # correlations_unordered = correlations.copy()
    # for i in range(correlations_unordered.shape[0]):
    #     soreted_vec = np.sort(correlations_unordered[i,:])
    #     correlations_unordered[i, :] = soreted_vec
    #
    # x = 0

    ###NEW ROADMAP

    rna_seq = pd.read_csv("/cs/zbio/jrosensk/road_map_rna_seq/read_count_matrix.gencode.tsv", sep="\t")
    # in_use_genes, rna_seq = get_sample_genes(rna_seq)

    gene_column_name = "gene_name"
    in_use_genes = set([x.split("(")[0].strip() for x in rna_seq[gene_column_name]])
    rna_seq[gene_column_name] = [x.split("(")[0].strip() for x in rna_seq[gene_column_name]]
    rna_seq = rna_seq.loc[rna_seq[gene_column_name].isin(set(features))]

    to_drop_ids = []
    seen_names = set()
    names = list(rna_seq[gene_column_name])
    ids = list(rna_seq.index)
    for cur_name, row_id in zip(names, ids):
        if cur_name in seen_names:
            to_drop_ids.append(row_id)
        else:
            seen_names.add(cur_name)
    rna_seq = rna_seq.loc[~rna_seq.index.isin(set(to_drop_ids))]
    # rna_seq = rna_seq[rna_seq.columns[1:]]

    # tissue_distances = {}
    features = set(features).intersection(set(rna_seq[gene_column_name]))
    sorted_features = sorted(list(features))
    gtex = rna_seq.set_index(gene_column_name).T
    gtex = gtex[sorted_features]
    gene_expression = gene_expression[sorted_features]
    logged_gtex = np.log2(np.array(gtex) + 1)
    logged_ccle = np.array(gene_expression)
    correlations = np.zeros([gene_expression.shape[0], gtex.shape[0]])
    # for i in range(gene_expression.shape[0]):
    #     for j in range(gtex.shape[0]):
    #         gene_expression_vec = logged_ccle[i, :]
    #         other_vec = logged_gtex[j, :]
    #         cur_pear, p_val = pearsonr(gene_expression_vec, other_vec)
    #         if abs(cur_pear) > 0.85:
    #             y = 0
    #         correlations[i,j] = cur_pear

    for i in range(gene_expression.shape[0]):
        for j in range(gtex.shape[0]):
            gene_expression_vec = logged_ccle[i, :]
            other_vec = logged_gtex[j, :]
            cur_pear, p_val = pearsonr(gene_expression_vec, other_vec)
            if abs(cur_pear) > 0.75:
                y = 0
            correlations[i, j] = cur_pear

    correlations_unordered = correlations.copy()
    for i in range(correlations_unordered.shape[0]):
        soreted_vec = np.sort(correlations_unordered[i, :])
        correlations_unordered[i, :] = soreted_vec

    x = 0

    # for gtex_tissue in rna_seq.columns[:-1]:
    #     distances_list = []
    #     cur_gtex_tissue = gtex.loc[gtex_tissue]
    #     cur_gtex_tissue = np.log2(np.array(cur_gtex_tissue) + 1)
    #     for achilles_tissue in tissue_set_achilles:
    #         achilles_cur_tissue_expression = gene_expression[gene_expression['tissue_types'] == achilles_tissue]
    #         achilles_cur_tissue_expression = achilles_cur_tissue_expression[sorted_features]
    #         achilles_cur_tissue_expression = np.array(achilles_cur_tissue_expression)
    #         avg_distance = calc_avg_distance_nearest_k(cur_gtex_tissue, achilles_cur_tissue_expression)
    #         el = (achilles_tissue, avg_distance)
    #         distances_list.append(el)
    #     tissue_distances[gtex_tissue] = distances_list
    x = 0

    ##################### NEW
    # gtex_info = pd.read_csv("/cs/staff/tommy/Work/GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    # tissue_col = "SMTS"
    # sample_id_col_name = "SAMPID"
    #
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
    #
    # rna_seq = pd.read_csv("gene_expression_tissue.descartes.csv")
    # rna_seq = rna_seq.fillna(0)
    # in_use_genes, rna_seq = get_sample_genes(rna_seq)
    # gene_column_name = "Name"
    # rna_seq = rna_seq.loc[rna_seq[gene_column_name].isin(set(features))]
    # rna_seq = rna_seq[rna_seq.columns[1:]]
    # descartes_data = rna_seq.set_index("Name").T
    #
    # tissue_distances = {}
    # features = set(features).intersection(set(descartes_data.columns))
    # sorted_features = sorted(list(features))
    # descartes_data = descartes_data[sorted_features]
    # for gtex_tissue, sample_ids in tissue_sample_dict.items():
    #     gtex = read_gtex_file(features, sample_ids)
    #     gtex = gtex.set_index("Description").T
    #     features = set(features).intersection(set(gtex.columns))
    #     sorted_features = sorted(list(features))
    #     gtex = gtex[sorted_features]
    #     cur_descartes_data = descartes_data[sorted_features]
    #     cur_gtex_tissue = np.log2(np.array(gtex) + 1)
    #     for descartes_tissue in rna_seq.columns[:-1]:
    #         cur_descartes_tissue = cur_descartes_data.loc[descartes_tissue]
    #         cur_descartes_tissue = np.log2(np.array(cur_descartes_tissue) + 1)
    #         avg_distance = calc_avg_distance_nearest_k(cur_descartes_tissue, cur_gtex_tissue)
    #         el = (gtex_tissue, avg_distance)
    #         if descartes_tissue in tissue_distances:
    #             cur_list = tissue_distances[descartes_tissue]
    #             cur_list.append(el)
    #             tissue_distances[descartes_tissue] = cur_list
    #         else:
    #             distances_list = []
    #             distances_list.append(el)
    #             tissue_distances[descartes_tissue] = distances_list
    x = 0

