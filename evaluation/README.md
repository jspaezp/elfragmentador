
# Evaluation data for ElFragmentador

This directory contains the workflow to evaluate the predictions of ElFragmentador.
It is meant to run on linux

## Run:

```shell
poetry run snakemake --verbose --cores 7 --directory $PWD -s external/snakefile.smk --configfile external/run.yml --rerun-incomplete

for i in evaluation/results/*/mokapot/mokapot.peptides.txt ; do  poetry run elfragmentador evaluate --input ${i} --nce 24,28,30,32,34,38,42 --out ${i}.evaluation.csv --assure_notrain 1 |& tee ${i}.evaluation.log ; done

for i in results/*/mokapot/*.csv ; do quarto render template.qmd -P csv_file:${PWD}/${i} --to=html --output "$(basename $(dirname $(dirname ${i}))).html" ; done
```


## Requirements

- Have comet in your path
- Have docker available

## TODO

- Move workflow to nextflow
- Move spectral library format to bibliospec
