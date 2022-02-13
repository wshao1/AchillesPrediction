import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from data_helper import clean_gene_names

if __name__ == '__main__':
    gene_effect_file = 'Achilles_gene_effect.csv'
    achilles_data = pd.read_csv(gene_effect_file)
    achilles_data = clean_gene_names(achilles_data, 'DepMap_ID')
    cell_ids = achilles_data['DepMap_ID']
    sample_info = pd.read_csv("sample_info.csv")
    tissue_types = []
    for cell_id in cell_ids:
        cur_tissue = list(sample_info[['DepMap_ID', 'sample_collection_site']][
                              sample_info.DepMap_ID == cell_id].sample_collection_site)[0]
        tissue_types.append(cur_tissue)
    achilles_data['tissue_types'] = tissue_types
    del achilles_data['DepMap_ID']
    grouped_by_tissue = achilles_data.groupby(by=["tissue_types"]).mean()
    tissue_names = []
    predictions = []
    target_gene = "VRK1"
    tissues = list(grouped_by_tissue[target_gene].index)
    preds = list(grouped_by_tissue[target_gene])

    fig, ax = plt.subplots()
    plt.bar(list(range(len(preds))), preds, width=0.8)
    ax.set_xticks(np.arange(len(preds)))
    ax.set_xticklabels(list(tissues), rotation=90)
    ax.set_title(target_gene)
    plt.show()
    x = 0