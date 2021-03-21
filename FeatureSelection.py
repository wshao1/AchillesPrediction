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

    # chi_squared = 0
    # for i in range(num_bins_A):
    #     for j in range(2):
    #         chi_squared += ((counts_array[i, j] - expected_array[i, j]) ** 2) / expected_array[i, j]
    chi_squared, p_val = chisquare(counts_array, f_exp=expected_array)
    return chi_squared, p_val


def get_top_k_columns_by_chi_squared(target, df, k):
    if False:  # os.path.isfile(file_name):
        chi_squared_list = joblib.load("file_name")
    else:
        chi_squared_list = []
        for idx, col_name in enumerate(df.columns):
            if idx > 0:
                mean_of_cur_gene = df[df.columns[idx]].mean()
                B = df[df.columns[idx]] > mean_of_cur_gene
                try:
                    chi_squared, p_val = chi_squared_statistic(target, B)
                except:
                    chi_squared, p_val = 0, 1
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
            if idx > 0: #TODO change this to use column name
                try:
                    # TODO handle nans better
                    corr, p_val = spearmanr(target, df[df.columns[idx]])
                except:
                    corr, p_val = 0, 1
                correlation_list.append((abs(corr), idx))
        # joblib.dump(correlation_list, file_name, 3)

    correlation_list = list(reversed(sorted(correlation_list, key=lambda x: x[0])))
    return [x[1] for x in correlation_list[0:k]]


def get_top_k_columns_by_correlation(target, df, k):
    if False:  # os.path.isfile(file_name):
        correlation_list = joblib.load(file_name)
    else:
        correlation_list = []
        for idx, col_name in enumerate(df.columns):
            if idx > 0:
                try:
                    #TODO handle nans better
                    corr, p_val = pearsonr(target, df[df.columns[idx]])
                except:
                    corr, p_val = 0, 1
                correlation_list.append((abs(corr), idx))
        # joblib.dump(correlation_list, file_name, 3)

    correlation_list = list(reversed(sorted(correlation_list, key=lambda x: x[0])))
    return [x[1] for x in correlation_list[0:k]]


def get_features(target, exrpession_data, num_features):
    res_chi = get_top_k_columns_by_chi_squared(target, exrpession_data, num_features)
    res_spear_corr = get_top_k_columns_by_spearman_correlation(target, exrpession_data, num_features)
    res_pear_corr = get_top_k_columns_by_correlation(target, exrpession_data, num_features)
    in_use_features = sorted(list(set(res_chi + res_spear_corr + res_pear_corr)))
    return in_use_features
