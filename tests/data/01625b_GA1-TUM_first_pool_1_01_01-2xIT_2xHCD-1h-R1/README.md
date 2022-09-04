# Intermediate files in a proteomics workflow for testing

## Generation

The original raw file was downloaded from PRIDE (prosit paper) and the corresponding fasta file was generated using the QC peptides for the run, the expected peptides for the "packet" and contaminants were appended (fasta bundled with maxquant).

Then the decoys were generated using pyteomics.

The included data is only a subset of the scans in the original file, generating using msconvert (proteowizard)

```shell
docker run \
    -v "$(pwd):/data" \
    -t chambm/pwiz-skyline-i-agree-to-the-vendor-licenses:x64 \
    wine msconvert \
    01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.mzML \
    -c pwiz_sample.config
```

Then the files were searched using comet

```shell
docker run \
    -v $PWD/:/data/\
    -it spctools/tpp \
    comet \
    -Pcomet.params.high_high \
    -DTUM_first_pool_1_contam.decoy.fasta \
    01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.mzML
```
