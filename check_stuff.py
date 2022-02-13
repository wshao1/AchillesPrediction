#!/usr/bin/python3 -u
import os
import subprocess

if __name__ == '__main__':
    # directory_in_str = "/cs/zbio/jrosensk/brain_rna_labelled"
    # directory = os.fsencode(directory_in_str)
    # file_ending_1 = "_RNA-Seq_1.fastq.gz"
    # file_ending_2 = "_RNA-Seq_2.fastq.gz"
    # alignment_ending = "_Aligned.out.sam"
    #
    # for file in os.listdir(directory):
    #	 filename = os.fsdecode(file)
    #	 if filename.endswith(file_ending_1):
    #		 base_name = filename[0:len(filename) - len(file_ending_1)]
    #		 aligned_file = base_name + alignment_ending
    #		 if not os.path.isfile(directory_in_str+"/"+aligned_file):
    #			 file_1 = directory_in_str+"/"+filename
    #			 file_2 = directory_in_str+"/"+ base_name + file_ending_2
    #			 cmd = f'sbatch --wrap="/cs/cbio/jon/bin/STAR-2.7.8a/source/STAR --genomeDir /cs/zbio/jrosensk/brain_rna_labelled/star_genome --readFilesIn {file_1} {file_2} --readFilesCommand zcat --runThreadN 16 --outFileNamePrefix {base_name}_" -c16 --mem=50g -t 0-1 -o slurm-STAR-map-{base_name}-%j.out'
    #			 subprocess.check_output(cmd, shell=True).decode()
    #			 x = 0

    directory_in_str = "/cs/zbio/jrosensk/brain_rna_labelled"
    directory = os.fsencode(directory_in_str)
    file_ending_1 = "_RNA-Seq_1.fastq.gz"
    file_ending_2 = "_RNA-Seq_2.fastq.gz"
    alignment_ending_prev = "_Aligned.out.sam"
    alignment_ending = "_Aligned.toTranscriptome.out.bam"
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(file_ending_1):
            base_name = filename[0:len(filename) - len(file_ending_1)]
            aligned_file = base_name + alignment_ending
            first_pass = base_name + alignment_ending_prev
            if (not os.path.isfile(directory_in_str + "/" + aligned_file)) and os.path.isfile(
                    directory_in_str + "/" + first_pass):
                if "SRR2557092" in base_name:
                    file_1 = directory_in_str + "/" + filename
                    file_2 = directory_in_str + "/" + base_name + file_ending_2
                    cmd = f'sbatch --wrap="/cs/cbio/jon/bin/STAR-2.7.8a/source/STAR --genomeDir star_genome --readFilesIn {file_1} {file_2} --quantMode TranscriptomeSAM --sjdbFileChrStartEnd {base_name}_SJ.out.tab --readFilesCommand zcat --runThreadN 16 --outFileNamePrefix {base_name}_2_" -c16 --mem=50g -t 0-24 -o slurm-STAR-map-{base_name}-%j.out'
                    subprocess.check_output(cmd, shell=True).decode()
                    x = 0
