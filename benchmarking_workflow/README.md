
Run as 

When using locally

```
$ snakemake --cores 1 --use-conda --conda-prefix /scratch/brown/jpaezpae/conda_envs/irt -s workflow.smk 
```

When using in slurm:

```
$ snakemake -j 500 --cluster-config cluster.json --cluster "sbatch -A {cluster.account} --mem={cluster.mem} -t {cluster.time} --cpus-per-task={cluster.cpus-per-task} -N {cluster.N}" --use-conda --conda-prefix /scratch/brown/jpaezpae/conda_envs/irt --local-cores 1 -s workflow.smk 
```
