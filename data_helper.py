import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import random
import argparse
import os


def get_intersection_gene_effect_expression_ids(achilles_data, expression_data):
    dep_map_id_achilles = achilles_data['ModelID']
    dep_map_id_gene_expression = expression_data['Unnamed: 0']
    return sorted(list(set(dep_map_id_achilles).intersection(set(dep_map_id_gene_expression))))


def create_train_test_df_using_manual_input(train_ids, test_ids):
    train_letters = ["train"] * len(train_ids)
    test_letters = ["test"] * len(test_ids)
    train_test_col = np.array(train_letters + test_letters)
    df = pd.DataFrame(train_test_col, columns=["train_test_split"])
    cur_col = np.array(train_ids + test_ids)
    df["id"] = cur_col
    return df


def create_train_test_df(ids_list, out_name="train_test_split.tsv", random_state_id=0, save_file=False):
    train_percentage = 0.75
    num_ids = len(ids_list)
    random.seed(random_state_id)
    random.shuffle(ids_list)
    train_ids_amount = int(train_percentage * num_ids)
    train_ids = sorted(ids_list[0:train_ids_amount])
    test_ids = sorted(ids_list[train_ids_amount:])
    df = create_train_test_df_using_manual_input(train_ids, test_ids)
    if save_file:
        df.to_csv(out_name, index=False, sep="\t")
    return df


def create_cv_folds_df(ids_list, num_folds=5, out_name="cross_validation_folds_ids.tsv", random_state_id=0,
                       save_file=False):
    ids_list = np.array(list(ids_list))
    cv_kf = KFold(n_splits=num_folds, random_state=random_state_id, shuffle=True)
    is_first = True
    index = 1
    for train_index, test_index in cv_kf.split(ids_list):
        X_train, X_test = ids_list[train_index], ids_list[test_index]
        if is_first:
            train_letters = ["train"] * len(train_index)
            test_letters = ["test"] * len(test_index)
            train_test_col = np.array(train_letters + test_letters)
            is_first = False
            folds_df = pd.DataFrame(train_test_col, columns=["state"])
        cur_col = np.concatenate([X_train, X_test])
        folds_df["fold_" + str(index)] = cur_col
        index += 1
    if save_file:
        folds_df.to_csv(out_name, index=False, sep="\t")
    return folds_df


def clean_gene_names(df, id_col_name):
    cur_cols = df.columns
    new_cols = []
    for col in cur_cols:
        if col == id_col_name:
            new_cols.append(col)
        else:
            new_col_name = col.split("(")[0].strip()
            new_cols.append(new_col_name)
    seen_names = set()
    new_new_cols = []
    for col in new_cols:
        if col in seen_names:
            tokens = col.split("_")
            if len(tokens) > 1:
                suffix = int(tokens[1])
                suffix += 1
                new_name = tokens[0] + "_" + str(suffix)
            else:
                new_name = col + "_1"
            seen_names.update(new_name)
        else:
            new_name = col
            seen_names.add(col)
        new_new_cols.append(new_name)
    df.columns = new_new_cols
    return df


def make_gene_file(gene_list, file_name):
    with open(file_name, 'w') as f_out:
        for gene in gene_list:
            f_out.write(gene + "\n")


def create_gene_list_files(achilles_df, out_dir='', num_files=100):
    all_genes = achilles_df.columns[1:]
    genes_per_file = int(len(all_genes) / num_files)
    cur_gene_list = []
    cur_index = 0
    cur_file_index = 0
    check_dir_exists_or_make(out_dir)
    while cur_index < len(all_genes):
        cur_gene_list.append(all_genes[cur_index])
        cur_index += 1
        if len(cur_gene_list) > genes_per_file:
            cur_file_name = out_dir + "gene_file_" + str(cur_file_index) + ".txt"
            cur_file_index += 1
            make_gene_file(cur_gene_list, cur_file_name)
            cur_gene_list = []


def check_dir_exists_or_make(name):
    if not os.path.exists(name):
        os.mkdir(name)


def get_intersecting_gene_ids_with_data_input(gene_expression, achilles_scores, achilles_id_col_name='ModelID',
                                       expression_id_col_name='Unnamed: 0', cv_df_file=None, train_test_df_file=None,
                                       should_clean_gene_names=True, num_folds=5):
    in_use_ids = get_intersection_gene_effect_expression_ids(achilles_scores, gene_expression)
    achilles_scores = achilles_scores.loc[achilles_scores[achilles_id_col_name].isin(in_use_ids)]
    gene_expression = gene_expression.loc[gene_expression[expression_id_col_name].isin(in_use_ids)]
    if should_clean_gene_names:
        achilles_scores = clean_gene_names(achilles_scores, achilles_id_col_name)
        gene_expression = clean_gene_names(gene_expression, expression_id_col_name)
    # gene_expression.to_csv("gene_expression_cell_lines_fixed.tsv", index=False, sep="\t")
    # achilles_scores.to_csv("achilles_effect_cell_lines_fixed.tsv", index=False, sep="\t")
    if train_test_df_file:
        train_test_df = pd.read_csv(train_test_df_file, sep="\t")
    else:
        train_test_df = create_train_test_df(in_use_ids)
    if cv_df_file:
        cv_df = pd.read_csv(cv_df_file, sep="\t")
    elif num_folds > 1:
        cv_df = create_cv_folds_df(in_use_ids, num_folds)
    else:
        cv_df = None
    return achilles_scores, gene_expression, train_test_df, cv_df


def get_intersecting_gene_ids_and_data(gene_effect_file, gene_expression_file, achilles_id_col_name='ModelID',
                                       expression_id_col_name='Unnamed: 0', cv_df_file=None, train_test_df_file=None,
                                       should_clean_gene_names=True, num_folds=5):
    ready_achilles_file = "ready_achilles.csv"
    ready_expression_file = "ready_gene_expression.csv"
    if os.path.isfile(ready_achilles_file) and os.path.isfile(ready_expression_file):
        gene_expression = pd.read_csv(ready_expression_file)#.sort_values(by=['Unnamed: 0'])
        achilles_scores = pd.read_csv(ready_achilles_file)#.sort_values(by=['ModelID'])
    else:
        # achilles_scores = pd.read_csv(gene_effect_file).dropna()
        # gene_expression = pd.read_csv(gene_expression_file)
        achilles_scores = gene_effect_file.dropna()
        gene_expression = gene_expression_file
        in_use_ids = get_intersection_gene_effect_expression_ids(achilles_scores, gene_expression)
        achilles_scores = achilles_scores.loc[achilles_scores[achilles_id_col_name].isin(in_use_ids)]
        gene_expression = gene_expression.loc[gene_expression[expression_id_col_name].isin(in_use_ids)]
        if should_clean_gene_names:
            achilles_scores = clean_gene_names(achilles_scores, achilles_id_col_name)
            gene_expression = clean_gene_names(gene_expression, expression_id_col_name)
        # achilles_scores.to_csv(ready_achilles_file, index=False)
        # gene_expression.to_csv(ready_expression_file, index=False)
    # gene_expression.to_csv("gene_expression_cell_lines_fixed.tsv", index=False, sep="\t")
    # achilles_scores.to_csv("achilles_effect_cell_lines_fixed.tsv", index=False, sep="\t")
    if train_test_df_file:
        train_test_df = pd.read_csv(train_test_df_file, sep="\t")
    else:
        in_use_ids = get_intersection_gene_effect_expression_ids(achilles_scores, gene_expression)
        train_test_df = create_train_test_df(in_use_ids)
    if cv_df_file:
        cv_df = pd.read_csv(cv_df_file, sep="\t")
    elif num_folds > 1:
        in_use_ids = get_intersection_gene_effect_expression_ids(achilles_scores, gene_expression)
        cv_df = create_cv_folds_df(in_use_ids, num_folds)
    else:
        cv_df = None
    return achilles_scores, gene_expression, train_test_df, cv_df


def get_tissue_types(expression_dat, sample_info_file="sample_info.csv"):
    sample_info = pd.read_csv(sample_info_file)
    tissue_types = []
    tissue_count = {}
    for cell_id in expression_dat['Unnamed: 0']:
        cur_tissue = list(sample_info[['ModelID', 'sample_collection_site']][
                              sample_info.ModelID == cell_id].sample_collection_site)[0]
        tissue_types.append(cur_tissue)
        if cur_tissue not in tissue_count:
            tissue_count[cur_tissue] = 1
        else:
            cur_count = tissue_count[cur_tissue]
            tissue_count[cur_tissue] = cur_count + 1
    return tissue_types, tissue_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_effect',
                        default='Achilles_gene_effect.csv')
    parser.add_argument('--gene_expression',
                        default='CCLE_expression.csv')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    achilles_scores = pd.read_csv(args.gene_effect)
    gene_expression = pd.read_csv(args.gene_expression)
    in_use_ids = get_intersection_gene_effect_expression_ids(achilles_scores, gene_expression)
    # del achilles_scores
    # del gene_expression
    create_train_test_df(in_use_ids, save_file=True)
    create_cv_folds_df(in_use_ids, save_file=True)
    create_gene_list_files(achilles_scores, "gene_files/")
