from snakemake.utils import min_version
min_version("6.0")

from contextlib import contextmanager
from loguru import logger as lg_logger

import tempfile
import requests
import re
from pprint import pformat as pfm

import numpy as np
import pandas as pd

lg_logger.info(f"Config: {pfm(config)}")

def split_configs(config: dict):
    """Split config into multiple configs.

    The config is split into 3 main configurations:
    1. The default configuration, which is applied to all the files if it is not
         overridden by the other configurations.
    2. The 'experiment' configuration, which is applied to all the files in the
         experiment.
         - The 'experiment' bundles files that share search parameters and database.
    3. The 'combine' configuration, which is applied to all the files in the
         combine.
         - The 'combine' bundles experiments that should be in the same spectral
           library.

    Returns:
        tuple[dict, dict, dict]: The default, experiment, and combine
            configurations.
    """
    DEFAULT_CONFIG = config["default"]
    combine_configs = {x["name"]: DEFAULT_CONFIG.copy() for x in config["combine"]}

    for x in config["combine"]:
        combine_configs[x["name"]].update(x)

    lg_logger.info(f"Combine Configs:\n {pfm(combine_configs)}")

    experiment_configs = {x["name"]: DEFAULT_CONFIG.copy() for x in config["experiments"]}
    for x in config["experiments"]:
        experiment_configs[x["name"]].update(x)

    lg_logger.info(f"Experiment Configs:\n {pfm(experiment_configs)}")
    return DEFAULT_CONFIG, combine_configs, experiment_configs


def expand_files(combine_configs, experiment_configs):
    """Expand the files in the combine and experiment configurations.

    Args:
        combine_configs (dict): The combine configurations.
        experiment_configs (dict): The experiment configurations.

    Returns:
        tuple[dict, dict]: The expanded combine and experiment configurations.
    """

    files = []
    exp_files_map = {x: {y:[] for y in ('mzML', 'pin')} for x in experiment_configs}

    for experiment, exp_values in experiment_configs.items():
        files.append(f"results/{experiment}/mokapot/mokapot.psms.txt")
        files.append(f"results/{experiment}/bibliospec/{experiment}.blib")
        files.append(f"results/{experiment}/spectrast.mokapot.psms.tsv")
        files.append(f"results/{experiment}/spectrast.mokapot.psms.sptxt")
        fasta_name = Path(exp_values['fasta']['value']).stem
        files.append(f"results/{experiment}/fasta/{fasta_name}.fasta")
        tmp_psm_files = exp_values["files"]
        tmp_psm_files = [Path(x).stem for x in tmp_psm_files]
        exp_files_map[experiment]['mzML'] = exp_values["files"]
        for raw_file in tmp_psm_files:
            pin_file = f"results/{experiment}/comet/{raw_file}.pin"
            exp_files_map[experiment]['pin'].append(pin_file)
            files.append(pin_file)


    lg_logger.info(f"Expanded files: {files}")
    for f in files:
        assert " " not in f, f"No spaces are allowed in file names. {f} has a space."
    return files, exp_files_map


DEFAULT_CONFIG, combine_configs, experiment_configs = split_configs(config)
files, exp_files_map = expand_files(combine_configs, experiment_configs)


rule all:
    input:
        *files

rule binaries:
    input:
        "bin/comet.exe",
        "bin/BlibBuild",
        "bin/BlibFilter",

from pathlib import Path

for exp_name in experiment_configs:
    for filetype in ['raw','mzml','comet','fasta','mokapot','bibliospec']:
        Path(f"results/{exp_name}/{filetype}").mkdir(parents=True, exist_ok=True)

# rule convert_raw:
#     """
#     Convert raw files to mzml
#
#     It is currently not implemented.
#     I am using docker to run msconvert in GCP
#     """
#     input:
#         "results/{experiment}/raw/{raw_file}",
#     output:
#         "results/{experiment}/mzml/{raw_file}.mzML"
#     run:
#         raise NotImplementedError
# ruleorder: link_mzML > convert_raw

def get_provided_file(wildcards):
    """Gets the location of the raw file from the
    configuration.
    """
    provided_files = experiment_configs[wildcards.experiment]['files']
    out = [x for x in provided_files if Path(x).stem == wildcards.raw_file]
    assert len(out) == 1, f"Could not find {wildcards.raw_file} in {provided_files}"
    if not out[0].endswith('.mzML'):
        raise NotImplementedError("Only mzML files are supported.")
    if not Path(out[0]).exists():
        lg_logger.warning(f"Provided file {out[0]} does not exist.")

    return out[0]

rule link_mzML:
    """
    Link mzML

    links an mzML from anywhere in the local computer to the mzml sub-directory
    """
    input:
        in_mzml = get_provided_file
    output:
        out_mzml = "results/{experiment}/mzml/{raw_file}.mzML"
    run:
        # The actual path for the link
        lg_logger.info("Linking mzML file")
        link = Path(output.out_mzml)

        lg_logger.info("Creaiting dir"+ f"mkdir -p {str(link.parent.resolve())}")
        # cmd = f"mkdir -p {str(link.parent.resolve())}"
        # shell(cmd)

        shell_cmd = f"ln -v -s  {str(Path(input.in_mzml).resolve())} {str(link.resolve())}"
        lg_logger.info(f"Shell Link ${shell_cmd}")
        shell(shell_cmd)
        lg_logger.info(f"Created link {link} to {input.in_mzml}")


rule get_fasta:
    """Gets fasta files needed for experiments

    Download the fasta file from the internet if its an internet location.
    It it exists locally it just copies it to the results folder
    """
    output:
        fasta_file = "results/{experiment}/fasta/{fasta_file}.fasta",
    run:
        fasta_conf = dict(experiment_configs[wildcards.experiment]['fasta'])
        lg_logger.info(f"Getting fasta file: {fasta_conf}")

        fasta_name = fasta_conf['value']
        fasta_type = fasta_conf['type']
        lg_logger.info(f"Getting fasta file: {fasta_conf['value']}")
        lg_logger.info(f"Getting fasta file: {fasta_conf['type']}")

        if fasta_type.startswith("url"):
            lg_logger.info("Fasta of type url")
            shell(f"wget {fasta_name} -O {output.fasta_file}")
        elif fasta_type.startswith("file"):
            lg_logger.info("Fasta of type file")
            shell(f"cp {fasta_name} {output.fasta_file}")
        elif fasta_type.startswith("uniprot"):
            lg_logger.info("Fasta of type uniprot")
            url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28proteome%3A{PROTEOME}%29%20AND%20%28reviewed%3Atrue%29"
            url = url.format(PROTEOME=fasta_name)
            lg_logger.info(url)
            lg_logger.info(f"Cache not found for {fasta_name}. Downloading from uniprot {url}.")
            all_fastas = requests.get(url).text

            with open(output.fasta_file, "w") as f:
                f.write(all_fastas)

        else:
            lg_logger.info(f"{fasta_type} not recognized as a fasta type")
            raise Exception(f"Unknown fasta type {fasta_type}")


def update_comet_config(infile, outfile, config_dict: dict):
    """
    Updates the values in a comet config file

    Reads the configuration in the infile and
    writes the updated configuration to the outfile
    """
    lg_logger.info(f"Updating comet file {infile} to {outfile}, with {config_dict}")
    with open(infile, "r") as f:
        with open(outfile, "w+", encoding="utf-8") as of:
            for line in f:
                for key in config_dict:
                    if line.startswith(key):
                        lg_logger.debug(str(line))
                        line = f"{key} = {config_dict[key]}\n"
                        lg_logger.debug(str(line))

                of.write(line)
    lg_logger.info(f"Done updating comet file {infile} to {outfile}, with {config_dict}")
    return outfile


rule generate_comet_config:
    """Generates comet config for each experiment

    Generates a comet config file by using the default config
    generated by `comet -p` and updating it with the values
    in the config.yml file
    """
    input:
        comet_exe = "bin/comet.exe"
    output:
        param_file = "results/{experiment}/{experiment}.comet.params",
    shadow: "minimal"
    run:
        lg_logger.info("Getting default comet params")
        shell(f"set -x; set -e ; set +o pipefail; out=$({input.comet_exe} -p) ; rc=$? ; echo 'Exit code for comet was ' $rc ; exit 0 ")
        lg_logger.info("Done running shell")
        comet_config = experiment_configs[wildcards.experiment]["comet"]
        lg_logger.info(f"Updating comet file comet.params.new to {output}, with {comet_config}")
        update_comet_config("comet.params.new", output.param_file, comet_config)
        lg_logger.info("Done updating expmt shell")


rule decoy_fasta:
  """Makes a fasta with decoys

  Generates a concatenated decoy database fasta file
  Currently is unused in this workflow, since it is handled directly by comet
  """
  input: "results/{experiment}/fasta/{fasta_file}.fasta"
  output: "results/{experiment}/fasta/{fasta_file}.decoy.fasta"
  run:
    pyteomics.fasta.write_decoy_db(
      source=None,
      output=None,
      mode='reverse',
      prefix="DECOY_",
      decoy_only=False)

"""
Future versions of comet might support this better.
Pre-making the index
rule comet_index:
  input:
    fasta = "results/{experiment}/fasta/{fasta_file}.fasta",
    comet_config = "comet_params/{experiment}.comet.params"
  output: "results/{experiment}/comet/{fasta_file}.fasta.idx"
  run:
    shell(f"comet -P{input.comet_config} -D{input.fasta}")
"""


rule comet_exe:
    """
    Downloads comet from github
    """
    output:
        "bin/comet.exe",
    params:
        source="https://github.com/UWPR/Comet/releases/latest/download/comet.linux.exe",
    run:
        shell("mkdir -p bin")
        shell("wget {params.source} -O {output}")
        shell("chmod +x {output}")

rule build_bibliospec:
    """
    Builds bibliospec and stores the binaries

    Maybe we shoudl contact the Brendan to have them make a release for linux ...
    """
    output:
        "bin/BlibBuild",
        "bin/BlibFilter",
        "bin/BlibSearch",
        "bin/BlibToMs2",
    threads: 8
    run:
        build_cmd = (
            "bash quickbuild.sh "
            f"-j{threads} --abbreviate-paths "
            "--hash optimization=space "
            "address-model=64 "
            "pwiz_tools/BiblioSpec "
            "toolset=gcc"
        )

        binaries = ["BlibBuild", "BlibFilter", "BlibSearch", "BlibToMs2"]

        shell("mkdir -p bin")
        shell("cd bin && if [ ! -d pwiz ] ;  then git clone --depth 1 https://github.com/ProteoWizard/pwiz.git ; fi")

        # Check if it has already been built
        blib_in_build = list(Path("bin/pwiz").rglob("*build*/*/*Blib*"))
        blib_maps = {k: [blib for blib in blibs if k in str(blib)] for k in binaries}

        if all([v for v in blib_maps.values()]):
          for k, v in blib_maps.items():
            shell(f"ln --verbose -s $PWD/{str(v[0])} bin/{k}")
        else:
            shell(f"cd bin/pwiz && {build_cmd} | tee buildlog.log")
            shell("cp pwiz/build-*/BiblioSpec/*.tar.bz2 bin/.")
            shell("tar -x --bzip2 -f bin/bibliospec-bin-*-*-release-*.tar.bz2")
            shell("cd pwiz && bash clean.sh")


def get_fasta_name(wildcards):
    """
    Gets the correct fasta name and path
    from the experiment config
    """
    fasta_name = Path(experiment_configs[wildcards.experiment]['fasta']['value']).stem
    out = f"results/{str(wildcards.experiment)}"
    out += f"/fasta/{str(fasta_name)}.fasta"
    return out

rule comet_search:
    """Uses comet to search a single mzml file

    Every run takes ~1.5 CPU hours, so in a 20 CPU machine, every file takes 5 minutes.
    Usually 16gb of memory is more than enough for any file.
    """
    input:
        fasta=get_fasta_name,
        mzml="results/{experiment}/mzml/{raw_file}.mzML",
        comet_config = "results/{experiment}/{experiment}.comet.params",
        comet_executable="bin/comet.exe",
    output:
        mzid_decoy="results/{experiment}/comet/{raw_file}.decoy.mzid",
        mzid="results/{experiment}/comet/{raw_file}.mzid",
        pepxml_decoy="results/{experiment}/comet/{raw_file}.decoy.pep.xml",
        pepxml="results/{experiment}/comet/{raw_file}.pep.xml",
        sqt_decoy="results/{experiment}/comet/{raw_file}.decoy.sqt",
        sqt="results/{experiment}/comet/{raw_file}.sqt",
        txt_decoy="results/{experiment}/comet/{raw_file}.decoy.txt",
        txt="results/{experiment}/comet/{raw_file}.txt",
        pin="results/{experiment}/comet/{raw_file}.pin",
    params:
        base_name="results/{experiment}/comet/{raw_file}",
    threads: 20
    run:
        lg_logger.info("Starting Comet")
        update_dict = { "num_threads": threads, "output_sqtfile": 0, "output_txtfile": 0, "output_pepxmlfile": 1, "output_mzidentmlfile": 0, "output_percolatorfile": 1, "decoy_search": 2 }
        handle, tmp_config = tempfile.mkstemp("config", dir = f"results/{wildcards.experiment}/comet/")

        lg_logger.info(f"Updating Params with: {update_dict}")
        update_comet_config(str(input.comet_config), str(tmp_config), update_dict)

        shell_cmd = f"{input.comet_executable} -P{tmp_config} -D{input.fasta} -N{params.base_name} {input.mzml}"
        lg_logger.info(f"Executing {shell_cmd}")
        shell(shell_cmd)


rule mokapot:
    """Runs mokapot to calculate the FDR

    It needs ~ 1gb of memory per file, being very generous.
    It is very fast, takes ~ 20 seconds per file.
    """
    input:
        pin_files = lambda x: exp_files_map[x.experiment]['pin'],
        fasta=get_fasta_name,
    output:
        decoy_peptides = "results/{experiment}/mokapot/mokapot.decoy.peptides.txt",
        # decoy_proteins = "results/{experiment}/mokapot/mokapot.decoy.proteins.txt",
        decoy_psms = "results/{experiment}/mokapot/mokapot.decoy.psms.txt",
        peptides = "results/{experiment}/mokapot/mokapot.peptides.txt",
        # proteins = "results/{experiment}/mokapot/mokapot.proteins.txt",
        psms = "results/{experiment}/mokapot/mokapot.psms.txt",
    params:
        out_dir = "",
    run:
        shell_cmd = [
            ' mokapot --keep_decoys ',
            #'--enzyme "[KR]" ',
            #f'--proteins {input.fasta} ',
            '--decoy_prefix "DECOY_" ',
        ]
        if len(input.pin_files) > 1:
            shell_cmd.append("--aggregate")

        shell_cmd += [
            f'--dest_dir results/{wildcards.experiment}/mokapot/ ',
            " ".join(input.pin_files),
        ]

        shell_cmd = " ".join(shell_cmd)
        lg_logger.info(f"Running: {shell_cmd}")
        shell(shell_cmd)


@contextmanager
def temporary_links(files, target_dir, clean = True):
    lg_logger.info(f"Linking {files} to {target_dir}")
    target_dir = Path(target_dir)
    linked_files = []
    for mzml in files:

        mzml = Path(mzml)
        assert mzml.exists(), f"Original file {mzml} does not exist"
        link = target_dir / Path(mzml).name
        if link.exists():
            lg_logger.debug(f"Link {link} already exists, skipping")
            continue
        else:
            lg_logger.debug(f"Copying {mzml} to {link}")
            # This is a hard link
            import shutil
            shutil.copy(str(mzml), str(link))
            if not link.exists():
                lg_logger.debug(f"Link {link} does not exist, check the original file ...")
            linked_files.append(link)
    try:
        yield linked_files
    finally:
        for link in linked_files:
            if link.exists():
                if clean:
                    link.unlink()
                    lg_logger.debug(f"Removed link {link}")


rule bibliospec:
    """Runs bibliospec to build the library

    Currently not implemented... because I cannot coimplile it in mac :(
    """
    input:
        blib_exe = "bin/BlibBuild",
        blib_filter = "bin/BlibFilter",
        psms = "results/{experiment}/mokapot/mokapot.psms.txt",
        mzML = lambda x: exp_files_map[x.experiment]['mzML']
    output:
        ssl_file = "results/{experiment}/bibliospec/{experiment}.ssl",
        library_name = "results/{experiment}/bibliospec/{experiment}.blib",
    run:
        shell(f"mkdir -p results/{wildcards.experiment}/bibliospec")
        lg_logger.info("Converting psms to ssl")
        convert_to_ssl(input.psms, output.ssl_file)
        lg_logger.info("Done Converting psms to ssl")
        # TODO implement multiprocessing of this section
        # Split ssl on x number of files that share peptides
        # then pass that to blibbuild
        # then merge the blibs

        shell_cmd = [
            f"{input.blib_exe}",
            "-C 2G", # minimum size to start caching
            f"-c {experiment_configs[wildcards.experiment]['BiblioSpec']['cutoff']}", # TODO refactor this ...
            "-m 500M", # sqlite cache size
            "-A", # warns ambiguous spectra
            str(output.ssl_file),
            str(output.library_name),
        ]
        shell_cmd = " ".join(shell_cmd)
        lg_logger.info("Prepping for bibliospec")
        # This creates soft links in the so bibliospec can find the raw spectra
        with temporary_links(files=input.mzML, target_dir=f"results/{wildcards.experiment}/bibliospec", clean=True) as linked_files:
            lg_logger.info(f"Running bibliospec: {shell_cmd}")
            import subprocess
            out = subprocess.run(shell_cmd, shell=True, check=False, capture_output=True)
            lg_logger.debug(f"Stdout: {out.stdout}")
            lg_logger.debug(f"Stderr: {out.stderr}")


rule bibliospec_filter:
    input:
        blib_filter = "bin/BlibFilter",
        library_name = "results/{experiment}/bibliospec/{experiment}.blib",
    output:
        filtered_library_name = "results/{experiment}/bibliospec/{experiment}.filtered.blib",
    run:
        shell_cmd = [
            f"{input.blib_filter}",
            "-f 0.01",
            "-m 500M",
            "-A",
            str(input.library_name),
            str(output.filtered_library_name),
        ]
        shell_cmd = " ".join(shell_cmd)
        lg_logger.info(f"Running bibliospec filter: {shell_cmd}")
        shell(shell_cmd)
# bin/BlibToMs2 --mz-precision 6 ./results/Firstexp/bibliospec/Firstexp.filter.blib

def convert_to_ssl(input_file, output_file):
    lg_logger.info(f"Reading File {input_file}")
    df = pd.read_csv(
        input_file,
        sep="\t",
        dtype={
            "SpecId": str,
            "Label": bool,
            "ScanNr": np.int32,
            "ExpMass": np.float16,
            "CalcMass": np.float16,
            "Peptide": str,
            "mokapot score": np.float16,
            "mokapot q-value": np.float32,
            "mokapot PEP": np.float32,
            "Proteins": str,
        },
    )

    lg_logger.info("Processing File")
    out_df = pd.DataFrame([_parse_line(x) for _, x in df.iterrows()])
    lg_logger.debug(f"{out_df}")
    lg_logger.info(f"Writting output: {output_file}")
    out_df.to_csv(output_file, sep="\t", index=False, header=True)
    lg_logger.info("Done")

SPEC_ID_REGEX = re.compile(r"^(.*)_(\d+)_(\d+)_(\d+)$")

def _parse_line(line):
    outs = SPEC_ID_REGEX.match(line.SpecId).groups()
    file_name, spec_number, charge, _ = outs

    first_period = 0
    last_period = 0
    for i, x in enumerate(line.Peptide):
        if x == ".":
            if first_period == 0:
                first_period = i
            else:
                last_period = i

    sequence = line.Peptide[first_period + 1 : last_period]

    line_out = {
        "file": Path(file_name).stem+".mzML",
        "scan": spec_number,
        "charge": charge,
        "sequence": sequence,
        "score-type": "PERCOLATOR QVALUE",
        "score": line["mokapot q-value"],
    }
    return line_out


def filter_mokapot_psm(psm_input: str, peptide_input: str):
    """
    psm_input = "mokapot/MannLabPhospho.mokapot.psms.txt"
    peptide_input = "mokapot/MannLabPhospho.mokapot.peptides.txt"
    filter_mokapot_psm(psm_input, peptide_input)
    """

    df = pd.read_table(str(psm_input))
    pep_df = pd.read_table(str(peptide_input))
    print(f"Filtering peptides for {str(psm_input)} {str(peptide_input)}")
    print(f"Pep df length {len(pep_df)}")
    pep_df = pep_df[(pep_df["mokapot q-value"] < 0.01) & (pep_df["mokapot PEP"] < 0.01)]
    passed_peptides = set(pep_df["Peptide"])
    print(f"Unique Peptides Passed {len(passed_peptides)}")

    if len(passed_peptides) == 0:
        raise ValueError(
            f"Numbber of peptides that passed is 0 for {psm_input} {peptide_input}"
        )

    print(f"Number of PSMs {len(df)}")
    df = df[(df["mokapot q-value"] < 0.01) & (df["mokapot PEP"] < 0.01)]

    df = df[[x in passed_peptides for x in df["Peptide"]]]
    print(f"Number of PSMs {len(df)} after peptide filtering")
    return df


def psm_df_to_tsv(df: pd.DataFrame, output: str):
    col_neworder = [
        "SpecId",
        "ScanNr",
        "Peptide",
        "mokapot PEP",
        "mokapot score",
        "Proteins",
    ]
    df["Peptide"] = [x.replace("[", "[+") for x in df["Peptide"]]
    df["Charge"] = ["_".join(line.split("_")[-2]) for line in df["SpecId"]]
    df["Peptide"] = [f"{x}/{y}" for x, y in zip(df["Peptide"], df["Charge"])]
    df["SpecId"] = ["_".join(line.split("_")[:-3]) for line in df["SpecId"]]
    df["mokapot PEP"] = 1 - df["mokapot PEP"]

    df = df[col_neworder]
    df.to_csv(str(output), sep="\t", index=False)
    print(df)


rule mokapot_spectrast_in:
    input:
        peptides = "results/{experiment}/mokapot/mokapot.peptides.txt",
        psms = "results/{experiment}/mokapot/mokapot.psms.txt",
    output:
        "results/{experiment}/spectrast.mokapot.psms.tsv",
    run:
        shell("mkdir -p spectrast_in")
        df = filter_mokapot_psm(input.psms, input.peptides)
        psm_df_to_tsv(df, str(output))


rule mokapot_spectrast:
    input:
        mokapot_in="results/{experiment}/spectrast.mokapot.psms.tsv",
    output:
        "results/{experiment}/spectrast.mokapot.psms.sptxt",
    shell:
        "set -x ; set -e ; "
        f"docker run -v ${PWD}/:/data/ spctools/tpp"
        " spectrast -V -cIHCD "
        " -Lresults/{wildcards.experiment}/spectrast.mokapot.psms.log"
        " -cNresults/{wildcards.experiment}/spectrast.mokapot.psms"
        " {input.mokapot_in} ;"
