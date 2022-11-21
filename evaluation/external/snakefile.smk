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

    Returns:
        tuple[dict, dict]: The default, experiment configurations.
    """
    DEFAULT_CONFIG = config["default"]
    experiment_configs = {x["name"]: DEFAULT_CONFIG.copy() for x in config["experiments"]}
    for x in config["experiments"]:
        experiment_configs[x["name"]].update(x)

    lg_logger.info(f"Experiment Configs:\n {pfm(experiment_configs)}")
    return DEFAULT_CONFIG, experiment_configs


def expand_files(experiment_configs):
    """Expand the files in the experiment configurations.

    Args:
        experiment_configs (dict): The experiment configurations.

    Returns:
        tuple[dict, dict]: The expanded experiment configurations.
    """

    files = []
    exp_files_map = {x: {y:[] for y in ('mzML', 'pin')} for x in experiment_configs}

    for experiment, exp_values in experiment_configs.items():
        files.append(f"results/{experiment}/mokapot/mokapot.psms.txt")
        files.append(f"results/{experiment}/bibliospec/{experiment}.blib")
        files.append(f"results/{experiment}/bibliospec/{experiment}.filtered.blib")
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


DEFAULT_CONFIG, experiment_configs = split_configs(config)
files, exp_files_map = expand_files(experiment_configs)


rule all:
    input:
        *files


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
    output:
        param_file = "results/{experiment}/{experiment}.comet.params",
    shadow: "minimal"
    run:
        lg_logger.info("Getting default comet params")
        shell(f"set -x; set -e ; set +o pipefail; out=$(comet -p) ; rc=$? ; echo 'Exit code for comet was ' $rc ; exit 0 ")
        lg_logger.info("Done running shell")
        comet_config = experiment_configs[wildcards.experiment]["comet"]
        lg_logger.info(f"Updating comet file comet.params.new to {output}, with {comet_config}")
        update_comet_config("comet.params.new", output.param_file, comet_config)
        lg_logger.info("Done updating expmt shell")



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
    output:
        pin="results/{experiment}/comet/{raw_file}.pin",
    params:
        base_name="results/{experiment}/comet/{raw_file}",
    threads: 20
    run:
        lg_logger.info("Starting Comet")
        update_dict = {
            "num_threads": threads,
            "output_sqtfile": 0,
            "output_txtfile": 0,
            "output_pepxmlfile": 0,
            "output_mzidentmlfile": 0,
            "output_percolatorfile": 1,
            "max_variable_mods_in_peptide":3,
            "num_output_lines": 2,
            "clip_nterm_methionine": 0,
            "decoy_search": 2 }
        handle, tmp_config = tempfile.mkstemp("config", dir = f"results/{wildcards.experiment}/comet/")

        lg_logger.info(f"Updating Params with: {update_dict}")
        update_comet_config(str(input.comet_config), str(tmp_config), update_dict)

        shell_cmd = f"comet -P{tmp_config} -D{input.fasta} -N{params.base_name} {input.mzml}"
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
    input:
        psms="results/{experiment}/mokapot/mokapot.psms.txt",
        peptides="results/{experiment}/mokapot/mokapot.peptides.txt",
        mzML=lambda x: exp_files_map[x.experiment]["mzML"],
    output:
        ssl_file="results/{experiment}/bibliospec/{experiment}.ssl",
        library_name="results/{experiment}/bibliospec/{experiment}.blib",
    run:
        shell(f"mkdir -p results/{wildcards.experiment}/bibliospec")
        lg_logger.info("Converting psms to ssl")
        cmd = f"python scripts/psms_to_ssl.py --input_file {input.psms} --input_peptides {input.peptides} --output_file {output.ssl_file}"
        shell(cmd)
        lg_logger.info("Done Converting psms to ssl")

        shell_cmd = [
            "docker run --user=$(id -u):$(id -g) -it --rm -v ${PWD}:/data:rw jspaezp/jspp_bibliospec BlibBuild",
            "-H " # more than one decimal for mods ...
            "-C 2G", # minimum size to start caching
            "-c 0.999",
            "-m 500M", # sqlite cache size
            "-A", # warns ambiguous spectra
            f"/data/{str(output.ssl_file)}",
            f"/data/{str(output.library_name)}",
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
        library_name = "results/{experiment}/bibliospec/{experiment}.blib",
    output:
        filtered_library_name = "results/{experiment}/bibliospec/{experiment}.filtered.blib",
    run:
        shell_cmd = [
            "docker run --user=$(id -u):$(id -g) -it --rm -v ${PWD}:/data:rw jspaezp/jspp_bibliospec BlibFilter ",
            f"/data/{str(input.library_name)}",
            f"/data/{str(output.filtered_library_name)}",
        ]
        shell_cmd = " ".join(shell_cmd)
        lg_logger.info(f"Running bibliospec filter: {shell_cmd}")
        import subprocess
        out = subprocess.run(shell_cmd, shell=True, check=False, capture_output=True)
        lg_logger.debug(f"Stdout: {out.stdout}")
        lg_logger.debug(f"Stderr: {out.stderr}")

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
