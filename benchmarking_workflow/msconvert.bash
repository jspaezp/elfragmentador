#!/usr/bin/zsh

set -x
set -e

if [[ ! -z "$(command -v docker)" ]] ; then
    echo "Docker found, running native docker"
    docker run \
        -it --rm -e WINEDEBUG=-all \
        -v $PWD/:/data chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert.exe \
        --zlib \
        --filter "peakPicking true 1-" \
        --filter "activation HCD" \
        --filter "analyzer FT" /data/$1
else
    echo "Docker not found, attempting singfularity run"
    singularity exec --writable \
        -S /mywineprefix/ ${CLUSTER_SCRATCH}/pwiz_sandbox/pwiz_sandbox \
        mywine msconvert \
        --zlib \
        --filter "peakPicking true 1-" \
        --filter "activation HCD" \
        --filter "analyzer FT" $1
fi

file_base="$(echo $1 | sed -e 's/.raw/.mzML/g'| xargs basename)"
cat ${file_base} > ./raw/${file_base}
rm -rf ${file_base}
ls -lcth ./raw/${file_base}
# touch "$(echo $1 | sed -e 's/.raw/.mzML/g')"
