from FeatureSelection import get_features
from data_helper import get_intersecting_gene_ids_and_data
import subprocess


def create_gene_bed_file(gene_names, target_gene_name, reference_path="/cs/cbio/jon/projects/PyCharmProjects/wgbs_tools/references/hg19/",
                    annotation_file="annotations.bed.gz"):
    regions = []
    for gene_name in gene_names:
        find_gene_names_cmd = "zcat {} | grep {}".format(reference_path + annotation_file, gene_name)
        result = subprocess.Popen(find_gene_names_cmd, shell=True, stdout=subprocess.PIPE)
        res = result.stdout.read().decode('utf-8').split("\n")
        is_first = True
        chrom = None
        start = None
        end = None
        for line in res:
            if len(line) > 0:
                line = line.strip()
                tokens = line.split("\t")
                cur_gene_name = tokens[-1]
                if cur_gene_name == gene_name and "promoter" in tokens[3]:
                    if is_first:
                        chrom = tokens[0]
                        start = tokens[1]
                        is_first = False
                    end = tokens[2]
        if chrom is not None:
            regions.append("{}\t{}\t{}\n".format(chrom, start, end))
    out_file_name = "{}_features.bed".format(target_gene_name)
    with open(out_file_name, "w") as f_out:
        for region in regions:
            f_out.write(region)
    x = 0


if __name__ == '__main__':
    target_gene_name = "RPP25L"
    achilles_scores, gene_expression, \
    train_test_df, cv_df = get_intersecting_gene_ids_and_data('Achilles_gene_effect.csv',
                                                              'CCLE_expression.csv',
                                                              cv_df_file="cross_validation_folds_ids.tsv",
                                                              train_test_df_file="train_test_split.tsv",
                                                              num_folds=1)
    achilles_scores = achilles_scores.sort_values(by=['DepMap_ID'])
    gene_expression = gene_expression.sort_values(by=['Unnamed: 0'])
    y = achilles_scores[target_gene_name]
    expression_feature_indices = get_features(y, gene_expression, 20)
    feature_names = gene_expression.columns[expression_feature_indices]
    feature_names = list(set(list(feature_names) + [target_gene_name]))
    print(feature_names)
    # feature_names = ["RPP25L", "RPP25", "VRK1", "VRK2"]
    # create_gene_bed_file(feature_names, target_gene_name)