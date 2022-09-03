#!/bin/bash

set -e
set -x

OUTPUT_DIR="."

for NOTEBOOK_NAME in *.qmd ; do
  quarto render ${NOTEBOOK_NAME} --output-dir ${OUTPUT_DIR}
done
