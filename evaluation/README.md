
# Evaluation data for ElFragmentador

This directory contains the workflow to evaluate the predictions of ElFragmentador.
It is meant to run on linux

## Run:

```shell
poetry run snakemake --verbose --cores 1 --directory $PWD -s snakefile.smk --configfile ./reference_files/run.yml --dry-run
```


## Requirements

- Have comet in your path
- Have docker available

## TODO

- Move workflow to nextflow
- Move spectral library format to bibliospec
