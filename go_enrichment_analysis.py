from goatools.go_enrichment import GOEnrichmentStudy
from goatools import obo_parser
import wget
import sys
import Bio.UniProt.GOA as GOA
import os
from ftplib import FTP
import gzip

from get_prot_id import get_prot_ids_of_genes

if __name__ == '__main__':
    data_folder = os.getcwd() + '/data'
    # arab_uri = '/pub/databases/GO/goa/ARABIDOPSIS/goa_arabidopsis.gaf.gz'
    arab_uri = '/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz'
    arab_fn = arab_uri.split('/')[-1]

    # Check if the file exists already
    arab_gaf = os.path.join(data_folder, arab_fn)
    if (not os.path.isfile(arab_gaf)):
        # Login to FTP server
        ebi_ftp = FTP('ftp.ebi.ac.uk')
        ebi_ftp.login()  # Logs in anonymously

        # Download
        with open(arab_gaf, 'wb') as arab_fp:
            ebi_ftp.retrbinary('RETR {}'.format(arab_uri), arab_fp.write)

        # Logout from FTP server
        ebi_ftp.quit()

    go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
    data_folder = os.getcwd() + '/data'

    # Check if we have the ./data directory already
    if (not os.path.isfile(data_folder)):
        # Emulate mkdir -p (no error if folder exists)
        try:
            os.mkdir(data_folder)
        except OSError as e:
            if (e.errno != 17):
                raise e
    else:
        raise Exception('Data path (' + data_folder + ') exists as a file. '
                                                      'Please rename, remove or change the desired location of the data path.')

    # Check if the file exists already
    if (not os.path.isfile(data_folder + '/go-basic.obo')):
        go_obo = wget.download(go_obo_url, data_folder + '/go-basic.obo')
    else:
        go_obo = data_folder + '/go-basic.obo'

    go = obo_parser.GODag(go_obo)

    methods = ["bonferroni", "fdr"]

    assoc = {}
    with gzip.open(arab_gaf, 'rt') as arab_gaf_fp:
        arab_funcs = {}  # Initialise the dictionary of functions

        # Iterate on each function using Bio.UniProt.GOA library.
        for entry in GOA.gafiterator(arab_gaf_fp):
            uniprot_id = entry.pop('DB_Object_ID')
            arab_funcs[uniprot_id] = entry

    pop = arab_funcs.keys()

    for x in arab_funcs:
        if x not in assoc:
            assoc[x] = set()
        assoc[x].add(str(arab_funcs[x]['GO_ID']))
    target_gene = ["DNAJC19"]
    gene_names = ['DLX6', 'MBTD1', 'TRHDE', 'NAALAD2', 'CD82', 'AURKA', 'TEKT2', 'PYCARD', 'TULP2', 'DLX5', 'QPCT', 'PCDH17', 'DNAJC15', 'CCRL2', 'CTCFL', 'EML2', 'RIPK3', 'ACY3', 'BTF3L4', 'MSI1', 'LACRT', 'SLC46A3', 'NOVA1', 'DMRTB1', 'ANKRD31', 'SDK1', 'NAPRT', 'CRB2', 'LRRC4C', 'CCDC3', 'DNAAF1', 'TEX264', 'EGFLAM', 'ID4', 'IGDCC3', 'SKIDA1', 'KCNIP4', 'SELENOV', 'DHRS4L2', 'RELN', 'OGDHL', 'LRCOL1', 'TMEFF1', 'MEX3A']
    target_gene_prot_ids = get_prot_ids_of_genes(target_gene)
    gene_prot_id_dict = get_prot_ids_of_genes(gene_names)
    target_study = list(target_gene_prot_ids.values())
    target_study = [x for inside_list in target_study for x in inside_list]
    study = list(gene_prot_id_dict.values())
    study = [x for inside_list in study for x in inside_list]

    g = GOEnrichmentStudy(pop, assoc, go,
                          propagate_counts=True,
                          alpha=0.05,
                          methods=methods)
    g_res_target = g.run_study(target_study)
    pathways_of_target = [x.GO for x in g_res_target if x.study_count > 0]
    target_assoc = {}
    for key, entry in assoc.items():
        for go_id in pathways_of_target:
            if go_id in entry:
                target_assoc[key] = entry
    g = GOEnrichmentStudy(pop, assoc, go,
                          propagate_counts=True,
                          alpha=0.05,
                          methods=methods)
    g_res = g.run_study(study)
    g.prt_txt(sys.stdout, g_res)
    x = 0