import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib
import warnings
from scipy.stats import pearsonr, spearmanr
from Models import process_for_training, train_no_eval
from configuration import gene_effect_file, gene_expression_file
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
import random
from data_helper import get_intersection_gene_effect_expression_ids, clean_gene_names, create_train_test_df, \
    get_intersecting_gene_ids_and_data


def get_percentile_binned_data(vec, num_bins=5):
    quartiles = []
    if len(vec) < num_bins:
        quartiles = vec
    else:
        for i in range(num_bins):
            cur_percent = float(i) / float(num_bins)
            list_index = cur_percent * len(vec)
            lower_index = int(np.floor(list_index))
            upper_index = lower_index + 1
            if upper_index < len(vec):
                upper_frac = np.ceil(list_index) - list_index
                lower_frac = 1.0 - upper_frac
                cur_percentil_val = (vec[lower_index] * lower_frac) + (
                            vec[upper_index] * upper_frac)
            else:
                cur_percentil_val = vec[lower_index]
            quartiles.append(cur_percentil_val)
    return quartiles


def correct(copy_numbers, essentiality_scores_list):
    scores_of_two = essentiality_scores_list[2]
    num_bins = 5
    cn2_percentile_binned = get_percentile_binned_data(scores_of_two, num_bins)
    for i in range(len(essentiality_scores_list)):
        if i != 2:
            cur_percentile_binned = get_percentile_binned_data(essentiality_scores_list[i], num_bins)
            if len(cur_percentile_binned) < len(cn2_percentile_binned):
                cn2_percentile_tmp = get_percentile_binned_data(scores_of_two, len(cur_percentile_binned))
                shifts = cn2_percentile_binned - cur_percentile_binned


def naive_correction(copy_numbers, essentiality_scores_list):
    avg_of_cn_2 = np.mean(essentiality_scores_list[2])
    cn_corrections = {}
    for idx, cn in enumerate(copy_numbers):
        if idx != 2:
            cur_mean = np.mean(essentiality_scores_list[idx])
            cur_mean = avg_of_cn_2 if np.isnan(cur_mean) else cur_mean
            cn_corrections[cn] = avg_of_cn_2 - cur_mean
    cn_corrections[copy_numbers[2]] = 0.0
    return cn_corrections


def gp_correction(copy_numbers, essentiality_scores_list):
    cn_corrections = {}
    x = []
    y = []
    for idx, cn in enumerate(copy_numbers):
        cur_essentialities = essentiality_scores_list[idx]
        x = x + [[cn]] * len(cur_essentialities)
        y = y + cur_essentialities
    y = np.array(y)
    X = np.array(x)
    kernel = RBF() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
    pred_2 = gpr.predict(np.array([np.array([2.0])])).flatten()[0]
    for idx, cn in enumerate(copy_numbers):
        if idx != 2:
            cur_pred = gpr.predict(np.array([np.array([cn])])).flatten()[0]
            cn_corrections[int(cn)] = pred_2 - cur_pred
    cn_corrections[copy_numbers[2]] = 0.0
    return cn_corrections


def plot_cn_essentiality(cn_cols, data_df, is_corrected, target_name):
    cn_hist = {}
    for gene_name in cn_cols:
        cn_gene_name = gene_name + "_x"
        achilles_gene_name = gene_name + "_y"
        cn_col = list(data_df[cn_gene_name])
        achilles_col = list(data_df[achilles_gene_name])
        for cn, essentiality in zip(cn_col, achilles_col):
            if cn in cn_hist:
                cur_list = cn_hist[cn]
                cur_list.append(essentiality)
            else:
                cur_list = [essentiality]
            cn_hist[cn] = cur_list

    cn_avg_score = [(cn, np.mean(scores_list)) for cn, scores_list in cn_hist.items()]
    cn_avg_score = sorted(cn_avg_score, key=lambda tup: tup[0])
    unzipped = list(zip(*cn_avg_score))
    cn_list = unzipped[0]
    scores_list = unzipped[1]

    sorted_list = [v for k, v in sorted(cn_hist.items(), key=lambda item: item[0])]
    cn_list = sorted(list(cn_hist.keys()))

    title_correction = "corrected " if is_corrected else ""

    fig, ax = plt.subplots()
    plt.bar(cn_list[0:30], scores_list[0:30], width=0.8)
    ax.set_xticks(np.arange(len(cn_list[0:30])))
    # ax.set_xticklabels(list(tissues), rotation=90)
    # ax.set_title(target_gene)
    ax.set_title("Copy Number vs Mean {}Essentiality {}".format(title_correction, target_name))
    plt.show()

    vec_list = []
    # total_list = range(int(cn_list[-1]))
    cur_index = 0
    cn_index = 0
    for cn in cn_list:
        while cur_index < cn:
            vec_list.append([])
            cur_index += 1
        vec_list.append(sorted_list[cn_index])
        cn_index += 1
        cur_index += 1

    fig, ax = plt.subplots()
    plt.boxplot(vec_list[0:30], widths=0.8, showfliers=False)
    ax.set_title("Copy Number vs {}Essentiality {}".format(title_correction, target_name))
    names = [str(x) for x in np.arange(len(cn_list[0:30]))]
    ax.set_xticklabels(names)
    plt.show()
    x = 0



def number_one_correction():
    achilles_id_col_name = 'DepMap_ID'
    cn_id_name = "Unnamed: 0"
    target_col = 'FAM50A (9130)'
    target_gene_name = 'ATP2A2'
    achilles_scores, gene_expression, \
    train_test_df, cv_df = get_intersecting_gene_ids_and_data('CRISPR_gene_effect.csv', 'CCLE_expression.csv',
                                       train_test_df_file="train_test_split.tsv", num_folds=0)
    copy_number_data = pd.read_csv("CCLE_gene_cn.csv")#, usecols=[cn_id_name, target_col])#.dropna()

    copy_number_data = clean_gene_names(copy_number_data, cn_id_name)
    copy_number_data = copy_number_data[[cn_id_name, target_gene_name]]
    dep_map_id_achilles = set(achilles_scores['DepMap_ID'])
    cn_ids = set(copy_number_data[cn_id_name])
    in_use_ids = dep_map_id_achilles.intersection(cn_ids)
    achilles_scores = achilles_scores.loc[achilles_scores[achilles_id_col_name].isin(in_use_ids)].sort_values(by=[achilles_id_col_name])
    gene_expression = gene_expression.loc[gene_expression[cn_id_name].isin(in_use_ids)].sort_values(by=['Unnamed: 0'])
    copy_number_data = copy_number_data.loc[copy_number_data[cn_id_name].isin(in_use_ids)].sort_values(by=['Unnamed: 0'])

    # in_genes = sorted(list(set(achilles_scores.columns).intersection(set(copy_number_data.columns))))
    # large_pos_corr = []
    # for gene in in_genes:
    #     essential = achilles_scores[gene]
    #     cn = copy_number_data[gene]
    #     corr, p_val = pearsonr(essential.values, cn.values)
    #     if corr >= 0.2:
    #         large_pos_corr.append(gene)
    # x = 0
    # copy_number_data = copy_number_data[[cn_id_name, target_gene_name]]
    tissue_specific_cn_analysis(gene_expression, achilles_scores, copy_number_data, target_gene_name)

    # cur_pear, p_val = pearsonr(copy_number_data[target_gene_name], achilles_scores[target_gene_name])
    # copy_number_data = copy_number_data[cn_id_name, target_col]
    model, features = train_no_eval(achilles_scores, gene_expression, target_gene_name,
                                    "linear", num_features=20, copy_number_data=copy_number_data)
    cn_effect = model.model.coef_[-1]
    achilles_scores[target_gene_name] = achilles_scores[target_gene_name].values - copy_number_data[target_gene_name].values * cn_effect
    model, features = train_no_eval(achilles_scores, gene_expression, target_gene_name,
                                    "linear", num_features=10, copy_number_data=None, should_plot=True,
                                    include_target_gene=True)

    x = 0


def tissue_specific_cn_analysis(expression_dat, achilles_effect, copy_number_data, target_gene_name):

    sample_info = pd.read_csv("sample_info.csv")
    tissue_types = []
    tissue_count = {}
    for cell_id in expression_dat['Unnamed: 0']:
        cur_tissue = list(sample_info[['DepMap_ID', 'sample_collection_site']][
                              sample_info.DepMap_ID == cell_id].sample_collection_site)[0]
        tissue_types.append(cur_tissue)
        if cur_tissue not in tissue_count:
            tissue_count[cur_tissue] = 1
        else:
            cur_count = tissue_count[cur_tissue]
            tissue_count[cur_tissue] = cur_count + 1
    expression_dat["tissue_types"] = tissue_types
    achilles_effect["tissue_types"] = tissue_types
    copy_number_data["tissue_types"] = tissue_types
    print("essentiality vs target expression")
    print(pearsonr(achilles_effect[target_gene_name].values, expression_dat[target_gene_name].values))
    print("target expression vs cn")
    print(pearsonr(expression_dat[target_gene_name].values, copy_number_data[target_gene_name].values))
    print("essentiality vs cn")
    print(pearsonr(achilles_effect[target_gene_name].values, copy_number_data[target_gene_name].values))
    old_achilles = achilles_effect.copy()
    large_tissues = [tiss for tiss, cnt in tissue_count.items() if cnt > 10]
    tissues_list = []
    # large_tissues = ['Colon']
    for tissue in large_tissues:
        cur_expression = expression_dat[expression_dat["tissue_types"] == tissue]
        cur_achilles = achilles_effect[achilles_effect["tissue_types"] == tissue]
        cur_cn_data = copy_number_data[copy_number_data["tissue_types"] == tissue]
        model, features = train_no_eval(cur_achilles.drop('tissue_types', 1), cur_expression.drop('tissue_types', 1), target_gene_name,
                                        "least_squares", num_features=20, copy_number_data=cur_cn_data.drop('tissue_types', 1))
        x_train = cur_expression[features[:-1]]
        copy_number_target = cur_cn_data[target_gene_name]
        x_train["copy_number"] = np.nan_to_num(copy_number_target, nan=np.median(copy_number_target.values))
        # before = x_train.copy()
        x_train = np.array(x_train)
        x_train = model.sclr.transform(x_train)
        cn_scaled = x_train[:, -1]
        cn_effect = model.model.coef_[-1]
        cur_achilles[target_gene_name] = cur_achilles[target_gene_name].values - cn_scaled * cn_effect

        print(f"before {tissue}")
        old_corr, p_val = pearsonr(achilles_effect[achilles_effect['tissue_types'] == tissue][target_gene_name].values, copy_number_data[copy_number_data['tissue_types'] == tissue][target_gene_name].values)
        print(old_corr)
        achilles_effect.loc[achilles_effect['tissue_types'] == tissue, target_gene_name] = cur_achilles[target_gene_name]

        if abs(cn_effect) > 0:
            corr, p_val = pearsonr(achilles_effect[achilles_effect['tissue_types'] == tissue][target_gene_name].values,
                                    copy_number_data[copy_number_data['tissue_types'] == tissue][
                                        target_gene_name].values)
            if abs(old_corr - corr) > 0.1 and old_corr > 0:
                tissues_list.append(tissue)
            print(cn_effect)
            print("after")
            print(corr)
        # new_achilles = achilles_effect[target_gene_name].values
        x = 0
    labels = []
    seq_of_data = []
    for tissue in large_tissues:
        cur_achilles = achilles_effect[achilles_effect["tissue_types"] == tissue][target_gene_name]
        seq_of_data.append(cur_achilles.values)
        
    plt.boxplot(seq_of_data, labels=large_tissues)
    plt.show()
    X = 0
        

    model, features = train_no_eval(old_achilles.drop('tissue_types', 1), expression_dat.drop('tissue_types', 1),
                                    target_gene_name,
                                    "least_squares", num_features=10, copy_number_data=None, should_plot=True,
                                    include_target_gene=True, tissues_list=tissues_list, header="uncorrected")
    model, features = train_no_eval(achilles_effect.drop('tissue_types', 1), expression_dat.drop('tissue_types', 1), target_gene_name,
                                    "least_squares", num_features=10, copy_number_data=None, should_plot=True,
                                    include_target_gene=True, tissues_list=tissues_list, header="corrected")
    # model, features = train_no_eval(old_achilles.drop('tissue_types', 1), expression_dat.drop('tissue_types', 1),
    #                                 target_gene_name,
    #                                 "xg_boost", num_features=10, copy_number_data=None, should_plot=True,
    #                                 include_target_gene=True, tissues_list=tissues_list, header="uncorrected")
    # model, features = train_no_eval(achilles_effect.drop('tissue_types', 1), expression_dat.drop('tissue_types', 1),
    #                                 target_gene_name,
    #                                 "xg_boost", num_features=10, copy_number_data=None, should_plot=True,
    #                                 include_target_gene=True, tissues_list=tissues_list, header="corrected")
    # new_achilles = achilles_effect[target_gene_name].values
    model, features = train_no_eval(achilles_effect.drop('tissue_types', 1), expression_dat.drop('tissue_types', 1),
                                    target_gene_name,
                                    "tree", num_features=10, copy_number_data=None, should_plot=True,
                                    include_target_gene=True, tissues_list=tissues_list)
    print(pearsonr(achilles_effect[target_gene_name].values, copy_number_data[target_gene_name].values))
    x=0


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        number_one_correction()
    achilles_id_col_name = 'DepMap_ID'
    target_col = 'FAM50A (9130)' #'VRK1 (7443)', 'BRAF (673)', 'NHLRC2 (374354)'#'VRK1 (7443)','SOX9 (6662)', 'PMM2 (5373)', 'PRKAR1A (5573)', 'A1BG (1)', 'A1CF (29974)', 'ABCA1 (19)', 'ABCF3 (55324)', 'MITF (4286)', 'FAM50A (9130)'
    target_name = "FAM50A"
    cn_id_name = "Unnamed: 0"
    achilles_data = pd.read_csv(gene_effect_file, usecols=[achilles_id_col_name, target_col]).dropna() #
    copy_number_data = pd.read_csv("CCLE_gene_cn.csv", usecols=[cn_id_name, target_col])#.dropna()
    # achilles_scores = clean_gene_names(achilles_data, achilles_id_col_name)
    copy_number_data = clean_gene_names(copy_number_data, cn_id_name)
    target_col = target_col.split("(")[0].strip()
    in_use_ids = get_intersection_gene_effect_expression_ids(achilles_data, copy_number_data)
    achilles_data = achilles_data.loc[achilles_data[achilles_id_col_name].isin(in_use_ids)]
    copy_number_data = copy_number_data.loc[copy_number_data[cn_id_name].isin(in_use_ids)]
    cn_id_name = achilles_id_col_name
    cn_cols = [cn_id_name] + list(copy_number_data.columns[1:])
    copy_number_data.columns = cn_cols
    achilles_cols_set = set(achilles_data.columns)
    cn_cols_set = set(cn_cols)
    intersecting_cols = achilles_cols_set.intersection(cn_cols_set)
    new_cols = [cn_id_name, target_col]# #[cn_id_name] + list(random.sample(list(intersecting_cols), 10000))
    achilles_data = achilles_data[new_cols]
    copy_number_data = copy_number_data[new_cols]
    copy_number_data = copy_number_data.set_index(cn_id_name)
    copy_number_data = (np.exp2(copy_number_data) - 1) * 2
    copy_number_data = np.round(copy_number_data)
    cn_cols = copy_number_data.columns
    achilles_data = achilles_data.set_index(achilles_id_col_name)
    data_df = copy_number_data.merge(achilles_data, on=achilles_id_col_name)
    del achilles_data
    del copy_number_data
    # has_enough_samples = []
    # for gene_name in cn_cols:
    #     cn_gene_name = gene_name + "_x"
    #     achilles_gene_name = gene_name + "_y"
    #     cn_col = list(data_df[cn_gene_name])
    #     achilles_col = list(data_df[achilles_gene_name])
    #     count_1 = 0
    #     cn_hist = {}
    #     for cn, essentiality in zip(cn_col, achilles_col):
    #         if cn == 1.0:
    #             count_1 += 1
    #         if cn in cn_hist:
    #             cur_count = cn_hist[cn]
    #             cur_count += 1
    #         else:
    #             cur_count = 1
    #         cn_hist[cn] = cur_count
    #     if count_1 >= 100:
    #         has_enough_samples.append(gene_name)
    cn_hist = {}
    for gene_name in cn_cols:
        cn_gene_name = gene_name + "_x"
        achilles_gene_name = gene_name + "_y"
        cn_col = list(data_df[cn_gene_name])
        achilles_col = list(data_df[achilles_gene_name])
        for cn, essentiality in zip(cn_col, achilles_col):
            if cn in cn_hist:
                cur_list = cn_hist[cn]
                cur_list.append(essentiality)
            else:
                cur_list = [essentiality]
            cn_hist[cn] = cur_list

    # cn_avg_score = [(cn, np.mean(scores_list)) for cn, scores_list in cn_hist.items()]
    # cn_avg_score = sorted(cn_avg_score, key=lambda tup: tup[0])
    # unzipped = list(zip(*cn_avg_score))
    # cn_list = unzipped[0]
    # scores_list = unzipped[1]

    sorted_list = [v for k, v in sorted(cn_hist.items(), key=lambda item: item[0])]
    cn_list = sorted(list(cn_hist.keys()))

    # fig, ax = plt.subplots()
    # plt.bar(cn_list[0:30], scores_list[0:30], width=0.8)
    # ax.set_xticks(np.arange(len(cn_list[0:30])))
    # # ax.set_xticklabels(list(tissues), rotation=90)
    # # ax.set_title(target_gene)
    # ax.set_title("Copy Number vs Mean Essentiality {}".format(target_name))
    # plt.show()

    vec_list = []
    # total_list = range(int(cn_list[-1]))
    cur_index = 0
    cn_index = 0
    for cn in cn_list:
        while cur_index < cn:
            vec_list.append([])
            cur_index += 1
        vec_list.append(sorted_list[cn_index])
        cn_index += 1
        cur_index += 1

    # fig, ax = plt.subplots()
    # plt.boxplot(vec_list[0:30], widths=0.8, showfliers=False)
    # ax.set_title("Copy Number vs Essentiality {}".format(target_name))
    # # ax.set_xticks(np.arange(len(sorted_list)))
    # plt.show()
    # plot_cn_essentiality(cn_cols, data_df, False, target_name)
    cn_corrections = naive_correction(list(range(len(vec_list))), vec_list)#
    data_df_corrrected = data_df.copy()
    data_df_corrrected[achilles_gene_name] = data_df_corrrected.apply(lambda row: row[achilles_gene_name] +
                                                            cn_corrections[int(row[cn_gene_name])], axis=1)
    # plot_cn_essentiality(cn_cols, data_df_corrrected, True, target_name)
    data_df_corrrected = data_df_corrrected[[target_col+"_y"]]
    data_df_corrrected.columns = [col_name.split("_")[0].strip() for col_name in list(data_df_corrrected.columns)]

    gene_expression = pd.read_csv(gene_expression_file)
    gene_expression = clean_gene_names(gene_expression, cn_id_name)
    data_df_corrrected = data_df_corrrected.reset_index()
    in_use_ids = get_intersection_gene_effect_expression_ids(data_df_corrrected, gene_expression)

    data_df_corrrected = data_df_corrrected.loc[data_df_corrrected[achilles_id_col_name].isin(in_use_ids)]
    gene_expression = gene_expression.loc[gene_expression["Unnamed: 0"].isin(in_use_ids)]
    train_test_df = create_train_test_df(in_use_ids)

    data_df_corrrected = data_df_corrrected.sort_values(by=['DepMap_ID'])
    gene_expression = gene_expression.sort_values(by=['Unnamed: 0'])
    c1_expression = gene_expression["VRK2"].values
    vrk1_scores = data_df_corrrected["VRK1"].values

    fig, ax = plt.subplots()
    plt.scatter(c1_expression, vrk1_scores)
    ax.set_title("VRK2 expression vs Essentiality scores of VRK1")
    plt.xlabel('VRK2 expression')
    plt.ylabel('Essentiality score VRK1')
    plt.show()
    x = 0

    target_gene_name, cv_rmse, cv_pearson, pearson_p_val, model, features, _ = process_for_training(data_df_corrrected, gene_expression, target_name, "xg_boost", train_test_df, None,
                         None,
                         1,
                         True, None,
                         False)
    joblib.dump(model, "cn_actually_corrected_vrk1_xgboost.pkl")
    x = 0

