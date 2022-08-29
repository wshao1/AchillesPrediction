import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import chisquare


def get_counts_bin(A, B, num_bins=6):
    # Assume B is array of booleans
    # Assume A is array of continuous numbers
    sorted_a = sorted(zip(A, B), key=lambda x: x[0])
    num_elements_per_bin = int(len(A) / num_bins)
    counts_array = np.zeros([num_bins, 2])
    num_elements_cur_bin = 0
    cur_bin = 0
    for a, b in sorted_a:
        if num_elements_cur_bin >= num_elements_per_bin and cur_bin < num_bins - 1:
            num_elements_cur_bin = 0
            cur_bin += 1
        num_elements_cur_bin += 1
        b_index = int(b)
        counts_array[cur_bin, b_index] += 1
    return counts_array


def chi_squared_statistic(A, B):
    counts_array = get_counts_bin(A, B)
    total = sum(sum(counts_array))
    num_bins_A = counts_array.shape[0]
    b_probs = np.zeros(counts_array.shape[1])
    for i in range(counts_array.shape[1]):
        b_probs[i] = sum(counts_array[:, i]) / total

    expected_array = np.zeros([num_bins_A, 2])
    for i in range(num_bins_A):
        for j in range(2):
            expected_array[i, j] = b_probs[j] * sum(counts_array[i, :])

    # chi_squared2 = 0
    # for i in range(num_bins_A):
    #     for j in range(2):
    #         chi_squared2 += ((counts_array[i, j] - expected_array[i, j]) ** 2) / expected_array[i, j]
    chi_squared, p_val = chisquare(counts_array.flatten(), f_exp=expected_array.flatten())
    return chi_squared, p_val


def filter_res_by_corr(df, k, in_list, corr_thresh=0.95):
    selected = set([in_list[0][1]])
    res_list = [in_list[0]]
    for val, idx in in_list:
        if idx not in selected:
            max_corr = 0
            min_p_val = 1
            for selected_idx in selected:
                corr, p_val = pearsonr(df[df.columns[selected_idx]], df[df.columns[idx]])
                if abs(corr) > max_corr:
                    max_corr = abs(corr)
                    min_p_val = p_val
            if max_corr < corr_thresh:
                selected.add(idx)
                res_list.append((val, idx))
        if len(res_list) > k:
            break
    return res_list


def get_top_k_columns_by_chi_squared(target, df, k):
    if False:  # os.path.isfile(file_name):
        chi_squared_list = joblib.load("file_name")
    else:
        chi_squared_list = []
        for idx, col_name in enumerate(df.columns):
            if df.columns[idx] != 'Unnamed: 0' and df.columns[idx] != 'DepMap_ID':
                mean_of_cur_gene = df[df.columns[idx]].mean()
                B = df[df.columns[idx]] > mean_of_cur_gene
                try:
                    chi_squared, p_val = chi_squared_statistic(target, B)
                except:
                    chi_squared, p_val = 0, 1
                # breaks_max = False
                # for _, el in chi_squared_list:
                #     corr_curr, p_val_2 = pearsonr(df[df.columns[idx]], df[df.columns[el]])
                #     if corr_curr > 0.9:
                #         breaks_max = True
                #         break
                # if not breaks_max:
                chi_squared_list.append((chi_squared, idx))
        # joblib.dump(chi_squared_list, file_name, 3)

    chi_squared_list = list(reversed(sorted(chi_squared_list, key=lambda x: x[0])))
    res_list = filter_res_by_corr(df, k, chi_squared_list)
    return [x[1] for x in res_list[0:k]]


def get_top_k_columns_by_chi_squared_cached(target, df, k, target_name, chi_squared_dict):
    if False:  # os.path.isfile(file_name):
        chi_squared_list = joblib.load("file_name")
    else:
        chi_squared_list = []
        for idx, col_name in enumerate(df.columns):
            dict_key_1 = f"{col_name}_{target_name}"
            dict_key_2 = f"{target_name}_{col_name}"
            if dict_key_1 not in chi_squared_dict and dict_key_2 not in chi_squared_dict:
                if df.columns[idx] != 'Unnamed: 0' and df.columns[idx] != 'DepMap_ID':
                    mean_of_cur_gene = df[df.columns[idx]].mean()
                    B = df[df.columns[idx]] > mean_of_cur_gene
                    try:
                        chi_squared, p_val = chi_squared_statistic(target, B)
                    except:
                        chi_squared, p_val = 0, 1
                    chi_squared_list.append((chi_squared, idx))
                    chi_squared_dict[dict_key_1] = chi_squared
            else:
                if dict_key_1 in chi_squared_dict:
                    chi_squared = chi_squared_dict[dict_key_1]
                    p_val = 1
                else:
                    chi_squared = chi_squared_dict[dict_key_2]
                chi_squared_list.append((chi_squared, idx))
        # joblib.dump(chi_squared_list, file_name, 3)

    chi_squared_list = list(reversed(sorted(chi_squared_list, key=lambda x: x[0])))
    return [x[1] for x in chi_squared_list[0:k]]


def get_top_k_columns_by_spearman_correlation(target, df, k):
    if False:  # os.path.isfile(file_name):
        correlation_list = joblib.load(file_name)
    else:
        correlation_list = []
        for idx, col_name in enumerate(df.columns):
            if df.columns[idx] != 'Unnamed: 0' and df.columns[idx] != 'DepMap_ID':  # TODO change this to use column name
                try:
                    # TODO handle nans better
                    corr, p_val = spearmanr(target, df[df.columns[idx]])
                except:
                    corr, p_val = 0, 1
                # breaks_max = False
                # for _, el in correlation_list:
                #     corr_curr, p_val_2 = pearsonr(df[df.columns[idx]], df[df.columns[el]])
                #     if corr_curr > 0.9:
                #         breaks_max = True
                #         break
                # if not breaks_max:
                correlation_list.append((abs(corr), idx))
        # joblib.dump(correlation_list, file_name, 3)

    correlation_list = list(reversed(sorted(correlation_list, key=lambda x: x[0])))
    res_list = filter_res_by_corr(df, k, correlation_list)
    return [x[1] for x in res_list[0:k]]


def get_top_k_columns_by_spearman_correlation_cached(target, df, k, target_name, spearman_dict):
    if False:  # os.path.isfile(file_name):
        correlation_list = joblib.load(file_name)
    else:
        correlation_list = []
        for idx, col_name in enumerate(df.columns):
            dict_key_1 = f"{col_name}_{target_name}"
            dict_key_2 = f"{target_name}_{col_name}"
            if dict_key_1 not in spearman_dict and dict_key_2 not in spearman_dict:
                if df.columns[idx] != 'Unnamed: 0' and df.columns[idx] != 'DepMap_ID':  # TODO change this to use column name
                    try:
                        # TODO handle nans better
                        corr, p_val = spearmanr(target, df[df.columns[idx]])
                    except:
                        corr, p_val = 0, 1
                    correlation_list.append((abs(corr), idx))
            else:
                if dict_key_1 in spearman_dict:
                    chi_squared = spearman_dict[dict_key_1]
                    p_val = 1
                else:
                    chi_squared = spearman_dict[dict_key_2]
                correlation_list.append((chi_squared, idx))
        # joblib.dump(correlation_list, file_name, 3)

    correlation_list = list(reversed(sorted(correlation_list, key=lambda x: x[0])))
    return [x[1] for x in correlation_list[0:k]]


def get_top_k_columns_by_correlation(target, df, k):
    if False:  # os.path.isfile(file_name):
        correlation_list = joblib.load(file_name)
    else:
        correlation_list = []
        for idx, col_name in enumerate(df.columns):
            if df.columns[idx] != 'Unnamed: 0' and df.columns[idx] != 'DepMap_ID':
                try:
                    # TODO handle nans better
                    corr, p_val = pearsonr(target, df[df.columns[idx]])
                except:
                    corr, p_val = 0, 1
                # breaks_max = False
                # for _, el in correlation_list:
                #     corr_curr, p_val_2 = pearsonr(df[df.columns[idx]], df[df.columns[el]])
                #     if corr_curr > 0.9:
                #         breaks_max = True
                #         break
                # if not breaks_max:
                correlation_list.append((abs(corr), idx))
        # joblib.dump(correlation_list, file_name, 3)

    correlation_list = list(reversed(sorted(correlation_list, key=lambda x: x[0])))
    res_list = filter_res_by_corr(df, k, correlation_list)
    return [x[1] for x in res_list[0:k]]


def get_top_k_columns_by_correlation_cached(target, df, k, target_name, pearson_dict):
    if False:  # os.path.isfile(file_name):
        correlation_list = joblib.load(file_name)
    else:
        correlation_list = []
        for idx, col_name in enumerate(df.columns):
            dict_key_1 = f"{col_name}_{target_name}"
            dict_key_2 = f"{target_name}_{col_name}"
            if dict_key_1 not in pearson_dict and dict_key_2 not in pearson_dict:
                if df.columns[idx] != 'Unnamed: 0' and df.columns[idx] != 'DepMap_ID':
                    try:
                        # TODO handle nans better
                        corr, p_val = pearsonr(target, df[df.columns[idx]])
                    except:
                        corr, p_val = 0, 1
                    correlation_list.append((abs(corr), idx))
            else:
                if dict_key_1 in pearson_dict:
                    chi_squared = pearson_dict[dict_key_1]
                    p_val = 1
                else:
                    chi_squared = pearson_dict[dict_key_2]
                correlation_list.append((chi_squared, idx))
        # joblib.dump(correlation_list, file_name, 3)

    correlation_list = list(reversed(sorted(correlation_list, key=lambda x: x[0])))
    return [x[1] for x in correlation_list[0:k]]


def get_features(target, exrpession_data, num_features, total_features=None):
    if num_features == 1:
        return list(get_top_k_columns_by_correlation(target, exrpession_data, num_features))
    else:
        res_chi = get_top_k_columns_by_chi_squared(target, exrpession_data, num_features)
        res_spear_corr = get_top_k_columns_by_spearman_correlation(target, exrpession_data, num_features)
        if total_features is not None:
            res_pear_corr = get_top_k_columns_by_correlation(target, exrpession_data, num_features+50)
            in_use_features = set(res_chi + res_spear_corr)
            cur_num_feats = len(in_use_features)
            added_features = set(res_pear_corr) - in_use_features
            to_take_feats = total_features - cur_num_feats
            assert(to_take_feats <= len(added_features))
            sorted_added_features_indices_to_take = [i for i, e in enumerate(res_pear_corr) if e in added_features][:to_take_feats] #TODO break after reaching correct number
            res_pear_corr = list(np.array(res_pear_corr)[sorted_added_features_indices_to_take])
        else:
            res_pear_corr = get_top_k_columns_by_correlation(target, exrpession_data, num_features)
        in_use_features = sorted(list(set(res_chi + res_spear_corr + res_pear_corr)))
        return in_use_features


def get_features_cached(target, exrpession_data, num_features, targe_name, chi_sqaured_dict, spearman_dict, pearson_dict):
    res_chi = get_top_k_columns_by_chi_squared_cached(target, exrpession_data, num_features, targe_name, chi_sqaured_dict)
    res_spear_corr = get_top_k_columns_by_spearman_correlation_cached(target, exrpession_data, num_features, targe_name, spearman_dict)
    res_pear_corr = get_top_k_columns_by_correlation_cached(target, exrpession_data, num_features, targe_name, pearson_dict)
    in_use_features = sorted(list(set(res_chi + res_spear_corr + res_pear_corr)))
    return in_use_features
