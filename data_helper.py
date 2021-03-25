import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import random
import argparse
import os


def get_intersection_gene_effect_expression_ids(achilles_data, expression_data):
    dep_map_id_achilles = achilles_data['DepMap_ID']
    dep_map_id_gene_expression = expression_data['Unnamed: 0']
    return sorted(list(set(dep_map_id_achilles).intersection(set(dep_map_id_gene_expression))))


def create_train_test_df(ids_list, out_name="train_test_split.tsv", random_state_id=0, save_file=False):
    train_percentage = 0.75
    num_ids = len(ids_list)
    random.seed(random_state_id)
    random.shuffle(ids_list)
    train_ids_amount = int(train_percentage * num_ids)
    train_ids = sorted(ids_list[0:train_ids_amount])
    test_ids = sorted(ids_list[train_ids_amount:])
    train_letters = ["train"] * len(train_ids)
    test_letters = ["test"] * len(test_ids)
    train_test_col = np.array(train_letters + test_letters)
    df = pd.DataFrame(train_test_col, columns=["train_test_split"])
    cur_col = np.array(train_ids + test_ids)
    df["id"] = cur_col
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


def get_intersecting_gene_ids_and_data(gene_effect_file, gene_expression_file, achilles_id_col_name='DepMap_ID',
                                       expression_id_col_name='Unnamed: 0', cv_df_file=None, train_test_df_file=None,
                                       should_clean_gene_names=True, num_folds=5):
    achilles_scores = pd.read_csv(gene_effect_file)
    gene_expression = pd.read_csv(gene_expression_file)
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
    else:
        cv_df = create_cv_folds_df(in_use_ids, num_folds)
    return achilles_scores, gene_expression, train_test_df, cv_df


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
