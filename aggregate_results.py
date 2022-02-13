import glob
import os
import numpy as np
import pandas as pd


if __name__ == '__main__':
    dir_prefix = "res"
    base_path = "/cs/cbio/jon/projects/PyCharmProjects/AchillesPrediction"
    dir_list = glob.iglob(os.path.join(base_path, "res*"))

    # dir_list_gene_list = glob.iglob(os.path.join(base_path + "/gene_files", "gene_file*"))
    orig_gene_list = []
    gene_file_index = 0
    while gene_file_index < 99:
        cur_file_name = base_path + "/gene_files/gene_file_{}.txt".format(str(gene_file_index))
        if os.path.isfile(cur_file_name):
            with open(cur_file_name, 'r') as g_file:
                for line in g_file:
                    line = line.split("(")[0].strip()
                    orig_gene_list.append(line)
        gene_file_index += 1

    res_df = pd.DataFrame(orig_gene_list, columns=["gene"])

    dir_list = list(dir_list)
    missing_value = -1.0
    for path in dir_list:
        gene_list = []
        rmse_list = []
        corr_list = []
        p_val_list = []
        cur_res_file_index = 0
        original_gene_index = 0
        if os.path.isdir(path):
            model_name = path.split("_")[1]
            while cur_res_file_index < 99:
                cur_file_name = os.path.join(path, "gene_file_{}.res.txt".format(str(cur_res_file_index)))
                if os.path.isfile(cur_file_name):
                    with open(cur_file_name, 'r') as f:
                        for line in f:
                            # if original_gene_index == 8008:
                            #     x = 0
                            line = line.rstrip()
                            tokens = line.split("\t")
                            # if tokens[0] == "KRT78":
                            #     x = 0
                            cur_original_gene = orig_gene_list[original_gene_index]
                            while cur_original_gene != tokens[0]:
                                gene_list.append(cur_original_gene)
                                rmse_list.append(missing_value)
                                corr_list.append(missing_value)
                                p_val_list.append(missing_value)
                                original_gene_index += 1
                                cur_original_gene = orig_gene_list[original_gene_index]
                            gene_list.append(tokens[0])
                            rmse_list.append(float(tokens[1]))
                            corr_list.append(float(tokens[2]))
                            p_val_list.append(float(tokens[3]))
                            original_gene_index += 1
                cur_res_file_index += 1
            cur_len = len(gene_list)
            if (orig_gene_list[0:cur_len] == gene_list):
                res_df[model_name + "_rmse"] = rmse_list + list(np.ones(len(orig_gene_list) - cur_len) * missing_value)
                res_df[model_name + "_corr"] = corr_list + list(np.ones(len(orig_gene_list) - cur_len) * missing_value)
                res_df[model_name + "_p_val"] = p_val_list + list(np.ones(len(orig_gene_list) - cur_len) * missing_value)
    cur_index = 0
    rmse_xgboost = res_df['xgboost_rmse']
    rmse_ens = res_df['ens_rmse']
    rmse_gp = res_df['gp_rmse']
    best = []
    while cur_index < res_df.shape[0]:
        cur_xgboost = rmse_xgboost[cur_index]
        cur_ens = rmse_ens[cur_index]
        cur_gp = rmse_gp[cur_index]
        if cur_gp < cur_ens:
            if cur_gp < cur_xgboost:
                best.append("gp")
            else:
                best.append("xgboost")
        else:
            if cur_ens < cur_xgboost:
                best.append("ens")
            else:
                best.append("xgboost")
        cur_index += 1
    res_df["best"] = best
    res_df.to_csv("results_table.tsv", index=False, sep='\t')