import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    res_file = '/cs/cbio/jon/projects/PyCharmProjects/AchillesPrediction/res/knn_25_k.predictions.txt'
    res_data = pd.read_csv(res_file, sep="\t")
    res_data = res_data.set_index("Gene_name")
    target_gene = "RPP25L"
    tissues = list(res_data.columns[1:])
    preds = list(res_data.loc[target_gene])[1:]

    fig, ax = plt.subplots()
    plt.bar(list(range(len(preds))), preds, width=0.8)
    ax.set_xticks(np.arange(len(preds)))
    ax.set_xticklabels(list(tissues), rotation=90)
    ax.set_title(target_gene)
    plt.show()
    x = 0
