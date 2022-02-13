import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import joblib
import os

from Models import EnsembleIDConverter


def group_list(preds_and_cells):
    cell_dict = {}
    for pred, cell in preds_and_cells:
        if cell in cell_dict:
            cell_preds = cell_dict[cell]
            cell_preds.append(pred)
        else:
            cell_preds = [pred]
        cell_dict[cell] = cell_preds

    cells = []
    avg_preds = []
    for cell in cell_dict.keys():
        cell_preds = cell_dict[cell]
        avg_pred = np.mean(cell_preds)
        cells.append(cell)
        avg_preds.append(avg_pred)
    return cells, avg_preds


if __name__ == '__main__':
    data_dir = '/cs/zbio/jrosensk/brain_rna_labelled/rsem_out'
    data_samples = []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            file_name = str(os.path.join(subdir, file))
            if (file_name.endswith("tsv")):
                basename = str(subdir).split("/")[-1].split("Homo")[0].split("yo")[-1]
                if basename.startswith("_"):
                    basename = basename[1:]
                if basename.endswith("_"):
                    basename = basename[:-1]
                split_under = basename.split("_")
                if len(split_under) > 2:
                    basename = "_".join([split_under[0], split_under[-1]])
                df = pd.read_csv(file_name, sep="\t")
                data_samples.append((df, basename))

    converter = EnsembleIDConverter()
    vrk1_features_non_corrected = set(['AFAP1', 'ANTXR1', 'ARMCX1', 'BIN2', 'C11orf68', 'CAP2', 'CC2D2A', 'CNRIP1', 'COL13A1', 'CORO2B',
                     'CXorf21', 'CYBB', 'DDR2', 'DENND1C', 'DENND2A', 'FHL1', 'GJC1', 'GPRASP2', 'GPSM1', 'KIRREL1',
                     'MAP1B', 'MAP4K1', 'MFAP4', 'NBEAL2', 'PCDHGC3', 'PCDHGC5', 'PFN2', 'PIP5K1B', 'PLEK', 'PTPN6',
                     'RBMS3', 'RUSC2', 'SPART', 'STXBP2', 'TCEAL7', 'THY1', 'TIMP2', 'TMC6', 'TMEM150B', 'VAMP8',
                     'VRK2'])
    vrk1_features_corrected = set(
        ['AFAP1', 'ANTXR1', 'ARMCX1', 'BIN2', 'C11orf68', 'CC2D2A', 'CNRIP1', 'COL13A1', 'CORO2B', 'CXorf21', 'CYP27C1',
         'DDR2', 'DENND1C', 'DENND2A', 'FHL1', 'GPRASP2', 'IRF8', 'KIRREL1', 'LOXL1', 'MAP1B', 'MAP4K1', 'PCDHGC3',
         'PCDHGC5', 'PFN2', 'PIP5K1B', 'PLEK', 'PTPN6', 'RBMS3', 'RNASET2', 'SPART', 'STXBP2', 'TCEAL7', 'TGFB1I1',
         'THY1', 'TIMP2', 'TMEM150B', 'VAMP8', 'VRK2'])
    new_data_samples = []
    xgb_model = joblib.load("cn_actually_corrected_vrk1_xgboost.pkl")
    for df, cell_type in data_samples:
        df[converter.gene_column_name] = df["gene_id"].apply(
            lambda x: converter.gene_id_to_name_lookup(x))
        df = df.loc[df[converter.gene_column_name].isin(vrk1_features_corrected)]
        df = df.sort_values(by=[converter.gene_column_name])
        input_vec = np.array([np.log2(np.array(df["TPM"]) + 1)])
        pred = xgb_model.predict(input_vec).flatten()
        new_data_samples.append((df, pred, cell_type))

    preds_and_cells = [(pred[0], cell) for df, pred, cell in new_data_samples]
    cells, avg_preds = group_list(preds_and_cells)
    x = 0

    # vrk2_expression_vs_predictions = [(df[df.Name == "C11orf68"].TPM.values[0], pred[0]) for df, pred in new_data_samples]
    # vrk2_expression = [a for a,b in vrk2_expression_vs_predictions]
    # preds = [b for a, b in vrk2_expression_vs_predictions]
    #
    #
    fig, ax = plt.subplots()
    plt.bar(cells, avg_preds)
    ax.set_xticklabels(cells, rotation=90)
    # ax.set_title(target_gene)
    ax.set_title("Avg model VRK1 essentiality prediction on brain samples")
    plt.xlabel('Cell type')
    plt.ylabel('Essentiality Prediction')
    plt.show()
    # data_samples = new_data_samples
    del new_data_samples
    x = 0