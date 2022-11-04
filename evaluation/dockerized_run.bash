#!/bin/bash

set -x
set -e

mkdir -p results
mkdir -p data

DOCKERFILE_TAG="dockerized_smk"

docker build . --tag ${DOCKERFILE_TAG} --file DOCKERFILE
# --no-cache

# --rm removes the container after it exits
# -v mounts a volume into the container
# --it makes the container interactive
time docker run -it --rm -v ${PWD}/external:/data/external -v ${PWD}/data:/data/data -v $PWD/bin:/data/bin -v $PWD/results:/data/results ${DOCKERFILE_TAG}
