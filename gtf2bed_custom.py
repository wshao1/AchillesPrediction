import gzip


def collapse_gtf(filename, outfile):
    fn_open = gzip.open if filename.endswith('.gz') else open
    gene_id = ""
    start = 0
    end = 0
    start_index = 3
    end_index = 4
    is_first = True
    with open(outfile, 'w') as f_out:
        with fn_open(filename) as fh:
            for line in fh:
                if filename.endswith('.gz'):
                    line = line.decode()
                if line.startswith('#'):
                    continue
                else:
                    tokens = line.split("\t")
                    info = tokens[-1]
                    info_tokens = info.split(";")
                    gene_id_list = info_tokens[0].split(" ")
                    assert (gene_id_list[0].strip() == "gene_id")
                    cur_gene_id = gene_id_list[1].strip()[1:-1]
                    cur_start = int(tokens[start_index])
                    cur_end = int(tokens[end_index])
                    if gene_id != cur_gene_id:
                        if is_first:
                            is_first = False
                        else:
                            final_line_tokens = []
                            final_line_tokens.append(tokens[0])
                            final_line_tokens.append(str(start))
                            final_line_tokens.append(str(end))
                            final_line_tokens.append(gene_id)
                            final_new_line = "\t".join(final_line_tokens)
                            f_out.write(final_new_line + "\n")
                        start = cur_start
                        end = cur_end
                        gene_id = cur_gene_id
                    else:
                        start = min(start, cur_start)
                        end = max(end, cur_end)


def create_id_name_map_from_gtf(filename, outfile):
    fn_open = gzip.open if filename.endswith('.gz') else open
    with open(outfile, 'w') as f_out:
        with fn_open(filename) as fh:
            for line in fh:
                if filename.endswith('.gz'):
                    line = line.decode()
                if line.startswith('#'):
                    continue
                else:
                    tokens = line.split("\t")
                    row_type = tokens[2]
                    if row_type == "gene":
                        info = tokens[-1]
                        info_tokens = info.split(";")
                        gene_id_list = info_tokens[0].split(" ")
                        assert (gene_id_list[0].strip() == "gene_id")
                        cur_gene_id = gene_id_list[1].strip()[1:-1]
                        for el in info_tokens:
                            if el.startswith(" "):
                                el = el[1:]
                            cur_tokens = el.split(" ")
                            if len(cur_tokens) > 0:
                                if(cur_tokens[0].strip() == "gene_name"):
                                    gene_name = cur_tokens[1].strip()[1:-1]
                                    f_out.write("{}\t{}\n".format(gene_name, cur_gene_id.split(".")[0]))
                                    break


if __name__ == '__main__':
    outfile = "/cs/zbio/jrosensk/ccle_fastq/reference_2/gencode.v29.primary_assembly.annotation.id_map.tsv"
    filename = "/cs/zbio/jrosensk/ccle_fastq/reference_2/gencode.v29.primary_assembly.annotation.gtf"
    create_id_name_map_from_gtf(filename, outfile)


