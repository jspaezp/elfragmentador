"""
It reads all the qc data and filters out psms that were detected too few times.
As well as peptides that have a high inter-quantile range in their retention time.

Note: the counts here might be decieving. Since the psms are first filtered by score
within raw files (keeping max 10/file) and then the median is taken over all the files.
So the counts are not the number of files in which the peptide was detected.
But will be in the range of 1*num_detected_files to 10*num_detected_files.
"""

import polars as pl


# pl.scan_parquet("prospect_data/Thermo_SRM_Pool_meta_data.parquet")
rt_qc = (
    pl.scan_parquet("prospect_data/*.parquet")
    .select(
        pl.col("modified_sequence"),
        pl.col("raw_file"),
        pl.col("andromeda_score"),
        pl.col("retention_time"),
        pl.col("indexed_retention_time"),
    )
    .select(
        pl.all()
        .sort_by(pl.col("andromeda_score"), descending=True)
        .over("modified_sequence")
    )
    .collect()
    .groupby(
        pl.col("modified_sequence"),
        pl.col("raw_file"),
    )
    .head(10)
    .groupby(
        pl.col("modified_sequence"),
    )
    .agg(
        pl.col("indexed_retention_time").median(),
        pl.col("retention_time").std().suffix("_sd"),
        pl.col("retention_time").mean().suffix("_mean"),
        pl.col("retention_time").median().suffix("median"),
        pl.col("retention_time").count().suffix("_c"),
        pl.col("retention_time").quantile(0.25).suffix("_q25"),
        pl.col("retention_time").quantile(0.75).suffix("_q75"),
    )
    .filter(pl.col("retention_time_c") > 3)
    .with_columns(iqr=pl.col("retention_time_q75") - pl.col("retention_time_q25"))
    .sort("iqr")
    .filter(pl.col("iqr") < 5)
)

rt_qc.write_parquet("processed_data/rt_qc.parquet")
print(rt_qc.shape)
print(rt_qc)
