import os


the_dict = [('Liver', 'ABHD5'),
('Skeletal Muscle', 'ACTA1'),
('Brain', 'ALB'),
('Heart', 'ANK2'),
('Brain', 'APP'),
('Brain', 'ATCAY'),
('Skin', 'ATP2A2'),
('Brain', 'AUH'),
('Heart', 'BRAF'),
('Brain', 'CACNA1A'),
('Heart', 'CACNA1C'),
('Skeletal Muscle', 'CACNA1S'),
('Heart, Skeletal Muscle', 'CAV3'),
('Brain', 'CHMP1A'),
('Skeletal Muscle', 'CHRNB1'),
('Skin', 'COL1A1'),
('Liver', 'CPS1'),
('Skeletal Muscle', 'CRYAB'),
('Heart', 'CSRP3'),
('Brain', 'CST3'),
('Brain', 'CTSD'),
('Heart, Skeletal Muscle', 'DES'),
('Heart, Skeletal Muscle', 'DMD'),
('Brain', 'DNAJC19'),
('Brain', 'DNAJC5'),
('Brain', 'DOCK6'),
('Skeletal Muscle', 'DYSF'),
('Skeletal Muscle', 'EGR2'),
('Heart', 'ENPP1'),
('Heart', 'FBN1'),
('Brain, Heart, Skin', 'FGFR2'),
('Brain, Heart', 'FLNA'),
('Heart', 'GATA4'),
('Heart', 'GJA1'),
('Heart', 'GNAI2'),
('Brain', 'GPSM2'),
('Brain', 'ITM2B'),
('Heart, Liver', 'JAG1'),
('Heart', 'JUP'),
('Heart', 'KCNJ2'),
('Brain, Skeletal Muscle', 'KIF21A'),
('Brain', 'KIF7'),
('Skin', 'KRT1'),
('Skin', 'KRT10'),
('Skin', 'KRT14'),
('Skin', 'KRT17'),
('Skin', 'KRT5'),
('Skeletal Muscle', 'LAMA2'),
('Skin', 'LAMA3'),
('Skin', 'LAMC2'),
('Heart, Skeletal Muscle, Skin', 'LMNA'),
('Skin', 'MITF'),
('Brain, Skeletal Muscle', 'MYH3'),
('Brain', 'MYO7A'),
('Skin', 'NAGA'),
('Brain', 'NHLRC2'),
('Skin', 'NOTCH3'),
('Thyroid', 'NTRK1'),
('Brain', 'OPHN1'),
('Brain', 'PMM2'),
('Heart', 'PRKAG2'),
('Heart', 'PRKAR1A'),
('Skin', 'PTCH1'),
('Skin', 'PTCH2'),
('Heart', 'PTPN11'),
('Brain', 'RPP25L'),
('Skeletal Muscle', 'RYR1'),
('Heart', 'RYR2'),
('Brain', 'SCN8A'),
('Heart, Skeletal Muscle', 'SGCD'),
('Skeletal Muscle', 'SLC25A4'),
('Brain', 'SLC6A8'),
('Skin', 'SNAI2'),
('Testis', 'SOX9'),
('Liver', 'TF'),
('Heart', 'TGFBR1'),
('Heart', 'TNNI3'),
('Brain', 'TOR1A'),
('Brain', 'TP53'),
('Heart', 'TPM1'),
('Skeletal Muscle', 'TPM2'),
('Skin', 'TYRP1'),
('Brain', 'VLDLR'),
('Brain', 'VRK1'),
('Brain', 'VRK2'),
('Heart', 'ZIC3')]


if __name__ == '__main__':
    new_dict = {}
    for key, entry in the_dict:
        new_dict[entry] = key
    dir_name = "/cs/zbio/jrosensk/essentiality_predictions"
    directory = os.fsencode(dir_name)
    out_file = "/cs/zbio/jrosensk/essentiality_predictions/all_preds.tsv"
    with open(out_file, 'w') as f_out:
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".preds.tsv"):
                with open(dir_name + "/" + filename) as cur_f:
                    for line in cur_f:
                        gene_name = line.split("\t")[0]
                        tissues = new_dict[gene_name]
                        f_out.write(line + "\t" + tissues + "\n")
                continue
            else:
                continue
