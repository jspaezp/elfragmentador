
import pandas as pd
import subprocess
import pathlib
import shutil

# samples = pd.read_table("sample_info.tsv").set_index("sample", drop=False)

samples = pd.DataFrame({
    'sample' : ["20161213_NGHF_DBJ_SA_Exp3A_HeLa_1ug_30min_15000_01", "20200604_RH_SVC_F10_Phos", "samp2"],
    'experiment' : ["second"] + ["first"] * 2,
    'comet_params': ["comet_params/comet.params.high_high"] + ["comet_params/comet_phospho.params.high_high"] * 2,
    'fasta': ["fasta/human_contam.fasta"] * 3,
    'server': ["ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2017/11/PXD006932"] + ["ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2020/11/PXD020183/"] + ["ftp://asdad"],
}).set_index("sample", drop=False)

def get_samples(experiment):
    return list(samples[samples['experiment'] == experiment]['sample'])

experiments = {k: v for k, v in zip(samples['experiment'], samples['comet_params'])}
samp_to_fasta = {k: v for k, v in zip(samples['sample'], samples['fasta'])}
samp_to_params = {k: v for k, v in zip(samples['sample'], samples['comet_params'])}
samp_to_ftp = {k: v for k, v in zip(samples['sample'], samples['server'])}

curr_dir = str(pathlib.Path(".").absolute())

if shutil.which("docker"):
    TPP_DOCKER=f"docker run -v {curr_dir}/:/data/ spctools/tpp "
else:
    TPP_DOCKER=f"singularity exec /scratch/brown/jpaezpae/opt/tpp.img"

rule all:
    input:
        # "samp1.decoy.pep.xml",
        "ip/second.iproph.pp.pep.xml",
        "spectrast/second.iproph.pp.sptxt",
        "spectrast/concensus_second.iproph.pp.sptxt",
        "ind_spectrast/20161213_NGHF_DBJ_SA_Exp3A_HeLa_1ug_30min_15000_01.pp.sptxt"

rule crap_fasta:
    output:
        "fasta/crap.fasta"
    shell:
        """
        mkdir -p fasta
        wget ftp://ftp.thegpm.org/fasta/cRAP/crap.fasta -O ./fasta/crap.fasta
        """

rule biognosys_irt_fasta:
    output:
        "fasta/irtfusion.fasta"
    shell:
        """
        mkdir -p fasta
        wget https://biognosys.com/media.ashx/irtfusion.fasta -O fasta/irtfusion.fasta
        """


rule human_fasta:
    output:
        "fasta/human.fasta"
    shell:
        """
        mkdir -p fasta
        wget https://www.uniprot.org/uniprot/\?query\=proteome:UP000005640%20reviewed:yes\&format\=fasta -O fasta/human.fasta
        """

rule contam_fasta:
    input:
        "fasta/crap.fasta",
        "fasta/human.fasta"
    output:
        "fasta/human_contam.fasta"
    shell:
        """
        cat fasta/human.fasta fasta/crap.fasta > fasta/human_contam.fasta
        """

rule download_file:
    output:
        "raw/{sample}.raw"
    run:
        server = samp_to_ftp[wildcards.sample]
        shell("mkdir -p raw")
        shell(f"wget {server}"+"/{wildcards.sample}.raw -O "+"{output}")
    

rule convert_file:
    input:
        "raw/{sample}.raw"
    output:
        "raw/{sample}.mzML"
    run:
        subprocess.run(['zsh', 'msconvert.bash', str(input)])

def get_fasta(wildcards):
    return samp_to_fasta[wildcards.sample]

def get_comet_params(wildcards):
    return samp_to_params[wildcards.sample]

rule comet_phospho_params:
    input:
        "comet_params/comet.params.high_high"
    output:
        "comet_params/comet_phospho.params.high_high"
    shell:
        """
        cat {input} | sed -e "s/variable_mod02 = 0.0 X 0 3 -1 0 0/variable_mod02 = 79.966331 STY 0 3 -1 0 0/g" > {output}
        """


rule comet_search:
    input:
        raw="raw/{sample}.mzML",
        fasta=get_fasta,
        comet_params=get_comet_params,
    output:
        decoy_pepxml = "comet/{sample}.decoy.pep.xml",
        forward_pepxml = "comet/{sample}.pep.xml"
    run:
        shell("mkdir -p comet")
        cmd = (
            f"{TPP_DOCKER}"
            f"comet -P{str(input.comet_params)} "
            f"-D{str(input.fasta)} "
            f"{str(input.raw)} " )

        print(cmd)
        shell(cmd)
        shell(f"cp raw/{wildcards.sample}.pep.xml ./comet/.")
        shell(f"cp raw/{wildcards.sample}.decoy.pep.xml ./comet/.")


rule interact:
    input:
        # rv_file="comet/{sample}.decoy.pep.xml",
        fw_file="comet/{sample}.pep.xml",
        fasta_file=get_fasta
    output:
        "pp/{sample}.pep.xml"
    shell:
        "set -x ; set -e ; mkdir -p pp ; "
        f"{TPP_DOCKER} "
        "xinteract -G -N{output} -nP "
        "{input.fw_file}"
        # "{input.rv_file} {input.fw_file}"

rule peptideprophet:
    input:
        "pp/{sample}.pep.xml"
    output:
        "pp/{sample}.pp.pep.xml"
    shell:
        """
        set -e
        set -x
        cp {input} {output}
        
        """ + 
        f"{TPP_DOCKER} " +
        "PeptideProphetParser {output} ACCMASS DECOY=DECOY_ DECOYPROBS NONPARAM"

rule indiv_spectrast:
    input:
        "pp/{sample}.pp.pep.xml"
    output:
        "ind_spectrast/{sample}.pp.sptxt",
        "ind_spectrast/{sample}.pp.splib",
        "ind_spectrast/{sample}.pp.pepidx",
        "ind_spectrast/{sample}.pp.spidx"
    shell:
        "set -x ; set -e ; mkdir -p ind_spectrast ; "
        f"{TPP_DOCKER}"
        " spectrast -c_RDYDECOY_ -cP0.9 -cq0.01 -cIHCD" # " -Mspectrast.usermods"
        " -Lind_spectrast/{wildcards.sample}.pp.log"
        " -cNind_spectrast/{wildcards.sample}.pp {input}"

def get_iproph_ins(wildcards):
    outs = expand("pp/{sample}.pp.pep.xml", sample = get_samples(wildcards.experiment))
    return outs

rule iprophet:
    input: get_iproph_ins
    output:
        "ip/{experiment}.iproph.pp.pep.xml"

    shell:
        "set -x ; set -e ; "
        f"{TPP_DOCKER}"
        "InterProphetParser THREADS=4 DECOY=DECOY NONSS NONSE"
        " {input} {output}"

# TODO check how to fix the issue where using interprophet destrys the model ...
rule spectrast:
    input:
        "ip/{experiment}.iproph.pp.pep.xml"
    output:
        "spectrast/{experiment}.iproph.pp.sptxt",
        "spectrast/{experiment}.iproph.pp.splib",
        "spectrast/{experiment}.iproph.pp.pepidx",
        "spectrast/{experiment}.iproph.pp.spidx",
        "spectrast/concensus_{experiment}.iproph.pp.sptxt",
        "spectrast/concensus_{experiment}.iproph.pp.splib",
        "spectrast/concensus_{experiment}.iproph.pp.pepidx",
        "spectrast/concensus_{experiment}.iproph.pp.spidx"
    shell:
        "set -x ; set -e ; mkdir -p spectrast ; "
        f"{TPP_DOCKER}"
        " spectrast -c_RDYDECOY_ -cP0.9 -cq0.01 -cIHCD" # " -Mspectrast.usermods"
        " -Lspectrast/{wildcards.experiment}.iproph.pp.log"
        " -cNspectrast/{wildcards.experiment}.iproph.pp {input} ;"
        f"{TPP_DOCKER}"
        " spectrast -cr2 -cAC -c_DIS " 
        " -Lspectrast/concensus_{wildcards.experiment}.iproph.pp.log"
        " -cNspectrast/concensus_{wildcards.experiment}.iproph.pp "
        " spectrast/{wildcards.experiment}.iproph.pp.splib"
