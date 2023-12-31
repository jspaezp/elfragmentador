#!/usr/bin/env python
# coding: utf-8

# .py file auto-generated from the .ipynb file
import duckdb
import traceback
import logging

duckdb.execute("SET enable_progress_bar = true;")
duckdb.execute("SET progress_bar_time = 200;")
duckdb.execute("SET temp_directory = 'temp.tmp';")

# Set up a logger to show some timing info
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)


def is_in_notebook():
    # https://stackoverflow.com/a/67622408
    rstk = traceback.extract_stack(limit=1)[0]
    return rstk[0].startswith("<ipython")


if is_in_notebook():
    get_ipython().system(' jupyter nbconvert --to script --no-prompt process_peaks.ipynb')


# This notebook compiles itself to a python script.
# the meant usage is to run it with a subset of the data
# and then commit the file for use on cloud training.

if is_in_notebook():
    PATH_PREFIX = "../prospect_data"
    con = duckdb.connect("prospect_dev.ddb")
else:
    import logging
    from fsspec import filesystem

    # this line will throw an exception if the appropriate filesystem interface is not installed
    duckdb.register_filesystem(filesystem("gcs"))
    logging.info("Running in cloud environment!!!!")
    con = duckdb.connect("prospect_prod.ddb")
    PATH_PREFIX = "gs://jspp_prospect_mirror"

METADATA_FILES = f"{PATH_PREFIX}/*meta_data.parquet"
ANNOTATION_FILES = f"{PATH_PREFIX}/*annotation.parquet"


if is_in_notebook():
    print(con.sql(f"SELECT * FROM parquet_scan('{METADATA_FILES}') LIMIT 5").to_df())


if is_in_notebook():
    print(con.sql(f"SELECT * FROM parquet_scan('{ANNOTATION_FILES}') LIMIT 5").to_df())


logging.info("Creating annotation table")
query = f"""
CREATE OR REPLACE TABLE rt_qc AS
SELECT
    modified_sequence,
    MEDIAN(retention_time) as retention_time_m,
    MEDIAN(indexed_retention_time) as irt_m,
    QUANTILE_CONT(retention_time, 0.25) as retention_time_q25,
    QUANTILE_CONT(retention_time, 0.75) as retention_time_q75,
    QUANTILE_CONT(indexed_retention_time, 0.25) as irt_q25,
    QUANTILE_CONT(indexed_retention_time, 0.75) as irt_q75,
    COUNT(*) as n
FROM
(
    SELECT modified_sequence, andromeda_score, retention_time, indexed_retention_time,
    FROM
    (
        SELECT 
            modified_sequence,
            andromeda_score,
            retention_time,
            indexed_retention_time,
            row_number() OVER (
                PARTITION BY modified_sequence, fragmentation, mass_analyzer
                ORDER BY andromeda_score DESC
            ) AS scan_rank
        FROM "{METADATA_FILES}"
    ) FILER WHERE scan_rank <= 20            -- take top 20 scans
)
GROUP BY modified_sequence
HAVING n >= 5 AND irt_q75 - irt_q25 < 10    -- require at least 5 scans and irt range < 10
"""

con.sql(query)


logging.info("Created annotation table")
con.sql("SELECT * FROM rt_qc")


logging.info("Creating filtered metadata table")
con.sql(
    f"""
    CREATE OR REPLACE TABLE filtered_meta AS
    SELECT
        raw_file,
        scan_number,
        modified_sequence,
        precursor_charge,
        precursor_intensity,
        mz,
        precursor_mz,
        fragmentation,
        mass_analyzer,
        retention_time,
        indexed_retention_time,
        orig_collision_energy,
        peptide_length,
        aligned_collision_energy,
        retention_time_m,
        irt_m
    FROM (
        SELECT *,
        row_number() OVER (
                        PARTITION BY
                            "{METADATA_FILES}".modified_sequence,
                            "{METADATA_FILES}".precursor_charge,
                            "{METADATA_FILES}".fragmentation,
                            "{METADATA_FILES}".mass_analyzer,
                            "{METADATA_FILES}".orig_collision_energy
                        ORDER BY ABS("{METADATA_FILES}".retention_time - rt_qc.retention_time_m) ASC
                    ) AS rt_diff_rank
        FROM "{METADATA_FILES}" 
        JOIN rt_qc ON ("{METADATA_FILES}".modified_sequence = rt_qc.modified_sequence)
        WHERE retention_time <= retention_time_q75 AND retention_time >= retention_time_q25
    )
    WHERE rt_diff_rank <= 10 AND peptide_length < 35
    """
)


logging.info("Created filtered metadata table")
con.sql("SELECT * FROM filtered_meta")


logging.info("Creating annotation table")
con.sql(
    f"""
        SELECT
            raw_file,
            scan_number,
            peptide_sequence,
            ion_type,
            no,
            charge,
            intensity,
            experimental_mass,
            theoretical_mass,
            fragment_score,
        FROM "{ANNOTATION_FILES}"
        WHERE neutral_loss = '' AND scan_number = 44454 AND ion_type = 'y' AND no = 3 AND charge = 2 AND peptide_sequence LIKE 'SHYVAQTGILWLLM%'
        GROUP BY raw_file, scan_number, peptide_sequence, ion_type, no, charge, intensity, experimental_mass, theoretical_mass, fragment_score
    """
).to_df()


# Note how multiple peaks can match a single fragment if there is ambiguity in the annotation
# This seems to happen in very rare cases and it cannot be solved by the fragment score.

# Just to get it over with I will use the max intensity one.

# I am using here a cherry-picked example that popped up during my qc of the data
# after the aggregation.


logging.info("Creating filtered annotations table")
con.sql(
    f"""
    CREATE OR REPLACE TABLE filtered_annotations AS
    SELECT
        raw_file,
        scan_number,
        peptide_sequence,
        precursor_charge,
        ion_type,
        no,
        charge,
        intensity,
        theoretical_mass as theoretical_fragment_mz,
        precursor_mz,
        fragmentation,
        mass_analyzer,
        retention_time,
        indexed_retention_time,
        orig_collision_energy,
        aligned_collision_energy,
    FROM (
        SELECT 
            filtered_meta.raw_file as raw_file,
            filtered_meta.scan_number as scan_number,
            peptide_sequence,
            precursor_charge,
            ion_type,
            no,
            charge,
            intensity,
            mz,
            theoretical_mass,
            precursor_mz,
            fragmentation,
            mass_analyzer,
            retention_time,
            indexed_retention_time,
            orig_collision_energy,
            aligned_collision_energy
        FROM
        (
            -- This section filters out neutral losses and keeps only the highest
            -- intensity fragment for each fragment type/position/charge combination
            SELECT * FROM
            (
                SELECT
                    *,
                    row_number() OVER (
                                    PARTITION BY
                                        peptide_sequence,
                                        scan_number,
                                        raw_file,
                                        ion_type,
                                        charge,
                                        no
                                    ORDER BY intensity DESC
                                ) AS fragment_rank
                FROM "{ANNOTATION_FILES}"
                WHERE neutral_loss = ''
            )
            WHERE fragment_rank = 1
        ) AS ranked_annots
        JOIN filtered_meta
        ON ranked_annots.peptide_sequence = filtered_meta.modified_sequence
        AND ranked_annots.scan_number = filtered_meta.scan_number
        AND ranked_annots.raw_file = filtered_meta.raw_file
    )
    GROUP BY ALL
    ORDER BY peptide_sequence, raw_file, scan_number
    """
)

con.sql("SELECT * FROM filtered_annotations LIMIT 10").to_df()


logging.info("Creating averaged annotations table")
con.sql(
    """
    CREATE OR REPLACE TABLE averaged_annotations AS
    SELECT
        LIST(raw_file),
        LIST(scan_number),
        filtered_annotations.peptide_sequence,
        precursor_charge,
        filtered_annotations.ion_type,
        filtered_annotations.no
            AS fragment_no,
        filtered_annotations.charge
            AS fragment_charge,
        SUM(filtered_annotations.intensity)/MAX(num_peptide_spec)
            AS fragment_intensity,
        MAX(filtered_annotations.intensity)
            AS max_fragment_intensity_prior_to_averaging,
        filtered_annotations.theoretical_fragment_mz,
        MEAN(filtered_annotations.precursor_mz)
            AS precursor_mz,
        filtered_annotations.fragmentation,
        filtered_annotations.mass_analyzer,
        MEAN(filtered_annotations.retention_time)
            AS retention_time,
        MEAN(indexed_retention_time)
            AS indexed_retention_time,
        filtered_annotations.orig_collision_energy,
        MEAN(aligned_collision_energy)
            AS aligned_collision_energy,
        num_peptide_spec,                                -- this is the number of spectra that were averaged
        COUNT(*) 
            AS num_averaged,
        COUNT(*) / MAX(num_peptide_spec) 
            AS pct_averaged -- the percentage of spectra that contained this fragment
    FROM (
        SELECT 
            peptide_sequence,
            mass_analyzer,
            fragmentation,
            orig_collision_energy,
            COUNT(DISTINCT scan_number || '::' || raw_file) AS num_peptide_spec
        FROM (
            SELECT DISTINCT
                raw_file,
                scan_number,
                peptide_sequence,
                mass_analyzer,
                fragmentation,
                orig_collision_energy
            FROM
                filtered_annotations
        ) AS inner_tmp
        GROUP BY peptide_sequence, mass_analyzer, fragmentation, orig_collision_energy
    ) AS middle_tmp
    JOIN filtered_annotations
    ON middle_tmp.peptide_sequence = filtered_annotations.peptide_sequence
    AND middle_tmp.mass_analyzer = filtered_annotations.mass_analyzer
    AND middle_tmp.fragmentation = filtered_annotations.fragmentation
    AND middle_tmp.orig_collision_energy = filtered_annotations.orig_collision_energy
    GROUP BY ALL
    ORDER BY filtered_annotations.peptide_sequence, filtered_annotations.no
    """
)


print(
    con.sql(
        """
    SELECT * from averaged_annotations ORDER BY pct_averaged DESC LIMIT 10
    """
    )
)

con.sql(
    """
SELECT * from averaged_annotations ORDER BY pct_averaged ASC LIMIT 10
"""
).to_df()


if is_in_notebook():
    from matplotlib import pyplot as plt

    df = con.sql(
        "SELECT peptide_sequence, MAX(pct_averaged) AS max_pct_avg, MAX(fragment_intensity) as max_intensity FROM averaged_annotations GROUP BY peptide_sequence"
    ).to_df()
    plt.hist(df["max_pct_avg"])
    plt.title("Distribution of max pct averaged")
    plt.show()

    plt.hist(df["max_intensity"])
    plt.title("Distribution of max fragment intensity")
    plt.show()


if is_in_notebook():
    from matplotlib import pyplot as plt

    df = con.sql(
        """
        SELECT * from averaged_annotations ORDER BY pct_averaged ASC LIMIT 100000
        """
    ).to_df()
    df
    plt.scatter(
        y=df["pct_averaged"],
        x=df["max_fragment_intensity_prior_to_averaging"],
        c=df["num_averaged"] / df["num_peptide_spec"],
        s=(df["num_averaged"] / df["num_peptide_spec"]) * 100,
    )
    plt.ylabel("Percetage of averaged spectra")
    plt.xlabel("max intensity pre-agg")
    plt.colorbar()
    # Save the plot to a file
    plt.savefig("peak_aggregation.png")
    plt.show()


if is_in_notebook():
    import numpy as np

    plt.scatter(
        y=df["pct_averaged"] + np.random.normal(0, 0.001, len(df)),
        x=df["max_fragment_intensity_prior_to_averaging"]
        + np.random.normal(0, 0.001, len(df)),
        s=1,
        alpha=0.05,
    )
    plt.ylabel("Percetage of averaged spectra")
    plt.xlabel("max intensity pre-agg")


logging.info("Creating nested annotations table")
con.sql(
    """
    CREATE OR REPLACE TABLE nested_annotations AS
    SELECT
        peptide_sequence,
        -- TODO: Add stripped peptide here
        precursor_charge,
        LIST (ion_type),
        LIST (fragment_no), -- TODO consider filter for this 
        LIST (fragment_charge),
        LIST (fragment_intensity),
        LIST (theoretical_fragment_mz),
        fragmentation,
        mass_analyzer,
        MEAN(retention_time) as retention_time,
        MEAN(indexed_retention_time) as indexed_retention_time,
        orig_collision_energy,
        MEAN(aligned_collision_energy) as aligned_collision_energy,
    FROM averaged_annotations
    WHERE 
        -- pretty liberal filter, I am assuming that peaks that were not averaged 
        -- are low abundance ... BUT who knows ...
        pct_averaged > 0.1 
    GROUP BY peptide_sequence, precursor_charge, fragmentation, mass_analyzer, orig_collision_energy
    """
)


con.sql("SELECT * FROM nested_annotations LIMIT 5").to_df()


"""
fragmentation	mass_analyzer	count_star()
0	HCD	FTMS	22066
1	CID	ITMS	4750
2	HCD	ITMS	4473
"""

counts = con.sql(
    "SELECT fragmentation, mass_analyzer, COUNT(*) FROM nested_annotations GROUP BY fragmentation, mass_analyzer"
).to_df()

logging.info("Counts per fragmentation and mass analyzer")
logging.info(counts)


# Export nested_annotations to parquet in bundles of ~20k
# Duckdb stores data in vectors of 2048 by default.
vectors_per_chunk = int(20_000 / 2048) + 1
index = 0

handle = con.execute("select * from nested_annotations")

while len(batch := handle.fetch_df_chunk(vectors_per_chunk)):
    batch.to_parquet(f"nested_annotations_{index}.parquet")
    if index == 0:
        print(batch.head(10))
    index += 1


# TODO add splitting
# TODO add split set
con.close()

