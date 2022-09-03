#!/bin/bash

set -e
set -x
# It is meant to be run from the root of the repository

OUTDIR=docs/notebooks
rm -rf ${OUTDIR}
mkdir -p ${OUTDIR}


for NOTEBOOK_NAME in $(find notebooks/docs -name '*.qmd'); do
   bn=$(basename $NOTEBOOK_NAME .qmd)
   poetry run quarto render ${NOTEBOOK_NAME} --to gfm --output docs/notebooks/${bn}.md $@

   # Manually move the images ... this should be fixed in a future release of quarto ...
   mv notebooks/docs/${bn}_files docs/notebooks/${bn}_files
done
