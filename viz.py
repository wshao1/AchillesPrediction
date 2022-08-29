import numpy as np
import matplotlib.pyplot as plt


def make_plot(y_train, y_test, train_pred, test_pred, gene_name, num_features, corr, p_val, model_name,
              target_essentiality, top_feat_expression):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(y_train, train_pred, s=10, c='b', marker="s", label='train')
    ax1.scatter(y_test, test_pred, s=10, c='r', marker="o", label='test')
    ymin, ymax = ax1.get_ylim()
    xmin, xmax = ax1.get_xlim()
    min_min = min(xmin, ymin)
    max_max = max(xmax, ymax)
    ax1.plot((min_min, max_max), (min_min, max_max), ls="--", c=".3")
    plt.legend(loc='lower right', prop={'size': 20})
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("{}".format(str(num_features) + " genes, non-linear"))

    ax1.title.set_fontsize(26)
    ax1.xaxis.label.set_fontsize(26)
    ax1.yaxis.label.set_fontsize(26)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(20)
    print("| corr: {}| pval: {}".format("{:.2f}".format(corr), format(p_val, '.2e')))
    ymin, ymax = ax1.get_ylim()
    ax1.set_yticks(np.round(np.linspace(ymin, ymax, 4), 2))
    plt.tight_layout()
    plt.show()