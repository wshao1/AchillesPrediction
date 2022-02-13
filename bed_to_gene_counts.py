import pandas as pd
import os

def process_mapped_reads_file_origi(line):
    tokens = line.split("\t")
    cur_gene_id = tokens[3]
    cur_gene_attributes = tokens[-1]
    cur_gene_name = cur_gene_attributes.split(";")[2].split(" ")[-1][1:-1]
    num_reads = len(cur_gene_attributes.split("|")[-1].split(";"))
    cur_entry = "{} ({})".format(cur_gene_name, cur_gene_id)
    return cur_gene_name, cur_gene_id, num_reads, cur_entry


def process_mapped_reads_file(line):
    tokens = line.split("\t")
    final_field = tokens[-1].split("|")
    cur_gene_id = final_field[0]
    cur_gene_attributes = final_field[1]
    num_reads = len(cur_gene_attributes.split(";"))
    # cur_entry = "{} ({})".format(cur_gene_name, cur_gene_id)
    return cur_gene_id, num_reads


if __name__ == '__main__':
    mapped_reads_suffix = ".mapped_reads.hg19.bed"#".collapsed.gencode.reads_mapped.bed"
    data_dir = "/cs/zbio/jrosensk/road_map_rna_seq_bed/"#"/cs/zbio/jrosensk/ccle_fastq/"#
    directory = os.fsencode(data_dir)
    seen_genes = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(mapped_reads_suffix):
            with open(data_dir + filename) as gene_reads_f:
                for line in gene_reads_f:
                    # cur_gene_name, cur_gene_id, num_reads, cur_entry = process_mapped_reads_file(line)
                    # if cur_entry not in seen_genes:
                    #     seen_genes[cur_entry] = 0
                    cur_gene_id, num_reads = process_mapped_reads_file(line)
                    if cur_gene_id not in seen_genes:
                        seen_genes[cur_gene_id] = 0
        else:
            continue

    samples_gene_counts = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(mapped_reads_suffix):
            gene_count_dict = dict(seen_genes)
            with open(data_dir + filename) as gene_reads_f:
                for line in gene_reads_f:
                    cur_gene_id, num_reads = process_mapped_reads_file(line)
                    gene_count_dict[cur_gene_id] = num_reads
            sample_name = ".".join(filename.split(".")[0:-2])
            samples_gene_counts[sample_name] = gene_count_dict
        else:
            continue

    sample_names = sorted(list(samples_gene_counts.keys()))
    sample_gene_counts_list = []
    sorted_keys = sorted(list(seen_genes.keys()))
    count_matrix = pd.DataFrame(columns=["gene_name"])
    count_matrix["gene_name"] = sorted_keys
    for sample_name in sample_names:
        cur_dict = samples_gene_counts[sample_name]
        cur_counts = [cur_dict[k] for k in sorted_keys]
        count_matrix[sample_name] = cur_counts

    count_matrix.to_csv("/cs/zbio/jrosensk/ccle_fastq/read_count_matrix.hg19.one_sample.tsv", index=False, sep="\t")
    x = 0

