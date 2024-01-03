#!/usr/bin/env python
# coding: utf-8

# .py file auto-generated from the .ipynb file
import duckdb
import traceback
import logging
import os
from pathlib import Path
from tqdm.auto import tqdm

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
    PATH_PREFIX = f"{os.environ['HOME']}/mount-folder/jspp_prospect_mirror/"

METADATA_FILES = f"{PATH_PREFIX}/*meta_data.parquet"
ANNOTATION_FILES = f"{PATH_PREFIX}/*annotation.parquet"
globbed_annotation_files = list(Path(PATH_PREFIX).rglob("*annotation.parquet"))

if len(globbed_annotation_files) == 0:
    raise ValueError("No annotation files found!")


if is_in_notebook():
    print(con.sql(f"SELECT * FROM parquet_scan('{METADATA_FILES}') LIMIT 5").to_df())


if is_in_notebook():
    print("Couns of metadata files by fragmentation and mass analyzer")
    print(
        con.sql(
            "SELECT COUNT(*), internal.fragmentation, internal.mass_analyzer "
            f"FROM (SELECT * FROM parquet_scan('{METADATA_FILES}')) AS internal "
            "GROUP BY internal.fragmentation, internal.mass_analyzer"
        ).to_df()
    )

    print("\n\n\n\n")
    print("Counts of unique neutral losses")
    print(
        con.sql(
            "SELECT COUNT(*), internal.neutral_loss "
            f"FROM (SELECT * FROM parquet_scan('{ANNOTATION_FILES}')) AS internal "
            "GROUP BY internal.neutral_loss "
            "ORDER BY COUNT(*) DESC "
        ).to_df()
    )


if is_in_notebook():
    print(con.sql(f"SELECT * FROM parquet_scan('{ANNOTATION_FILES}') LIMIT 5").to_df())


logging.info("Defining types")
try:
    con.execute(
        """
        CREATE TYPE ion_type_type AS ENUM ('a', 'b', 'c', 'x', 'y', 'z');
        CREATE TYPE fragmentation_type AS ENUM ('HCD', 'CID', 'ETD');
        CREATE TYPE mass_analyzer_type AS ENUM ('ITMS', 'FTMS');
        """
    )
except duckdb.CatalogException:
    logging.info("Types already defined")
logging.info("Done defining types")


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

# Create indices
logging.info("Creating indices")
con.sql("CREATE INDEX modseq_idx ON rt_qc(modified_sequence)")
logging.info("Created indices")

glimpse = con.sql("SELECT * FROM rt_qc LIMIT 5")
if is_in_notebook():
    print(glimpse.to_df())


logging.info("Creating filtered metadata table")
con.sql(
    f"""
    CREATE OR REPLACE TABLE filtered_meta AS
    SELECT
        raw_file,
        scan_number,
        modified_sequence,
        CAST (precursor_charge AS TINYINT) AS precursor_charge,
        precursor_intensity,
        mz,
        precursor_mz,
        CAST (fragmentation AS fragmentation_type) AS fragmentation,
        CAST (mass_analyzer AS mass_analyzer_type) AS mass_analyzer,
        retention_time,
        indexed_retention_time,
        orig_collision_energy,
        CAST (peptide_length AS TINYINT) AS peptide_length,
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

glimpse = con.sql("SELECT * FROM filtered_meta LIMIT 5")
count = con.sql("SELECT COUNT(*) FROM filtered_meta")

logging.info(f"Filtered metadata table has {count.to_df().iloc[0,0]} rows")
glimpse.to_df()


# Create indices
logging.info("Creating indices")
con.sql(
    "DROP INDEX IF EXISTS modseq_fmeta_idx ; CREATE INDEX modseq_fmeta_idx ON filtered_meta(modified_sequence)"
)
logging.info("Created indices")


if is_in_notebook():
    con.sql(
        f"""
            SELECT
                raw_file,
                scan_number,
                peptide_sequence,
                CAST (ion_type as ion_type_type) AS ion_type,
                CAST (no AS TINYINT) AS no,
                CAST (charge AS TINYINT) AS charge,
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


# NOTE!!! Here there isn un-handled ambiguity in the annotation. ...
# there can be multiple peaks that match a single ion type, no, charge, and sequence.

logging.info("Creating local annotation table")


# Create the table with the first file
first_annot_file = str(globbed_annotation_files[0])
logging.info(f"Creating local annotation table from {first_annot_file}")

# read_parquet('test.parq')

con.sql(
    f"""
        CREATE OR REPLACE TABLE tmp AS SELECT
            raw_file,
            scan_number,
            peptide_sequence,
            CAST (ion_type as ion_type_type) AS ion_type,
            CAST (no AS TINYINT) AS fragment_no,
            CAST (charge AS TINYINT) AS charge,
            neutral_loss,
            intensity,
            theoretical_mass as theoretical_fragment_mz,
            fragment_score,
        FROM read_parquet('{first_annot_file}', filename = false)
        WHERE neutral_loss = '' AND fragment_score > 0.5 AND ion_type IN ('a', 'b', 'c', 'x', 'y', 'z');
        """
)

out = con.sql(
    f"""
        EXPLAIN ANALYZE
        CREATE OR REPLACE TABLE local_annotations AS
        SELECT
            filtered_meta.raw_file as raw_file,
            filtered_meta.scan_number as scan_number,
            peptide_sequence,
            CAST (precursor_charge AS TINYINT) AS precursor_charge,
            CAST (ion_type as ion_type_type) AS ion_type,
            fragment_no,
            neutral_loss,
            CAST (charge AS TINYINT) AS charge,
            intensity,
            -- mz, -- This is the observd mz
            theoretical_fragment_mz,
            precursor_mz, -- "same" as mz
            CAST (fragmentation AS fragmentation_type) AS fragmentation,
            CAST (mass_analyzer AS mass_analyzer_type) AS mass_analyzer,
            retention_time,
            indexed_retention_time,
            orig_collision_energy,
            aligned_collision_energy
        FROM tmp
        INNER JOIN filtered_meta
        ON tmp.peptide_sequence = filtered_meta.modified_sequence
        AND tmp.scan_number = filtered_meta.scan_number
        AND tmp.raw_file = filtered_meta.raw_file
        -- ORDER BY peptide_sequence DESC; -- 3 min with sorting, 11s without
    """
)
print(out["explain_value"].to_df().iloc[0, 0])

# Append the rest of the files
chunksize = 50
rest_files = globbed_annotation_files[1:]
chunked = [rest_files[i : i + chunksize] for i in range(0, len(rest_files), chunksize)]
for annot_file_chunk in tqdm(chunked):
    # SELECT * FROM read_parquet(['file1.parquet', 'file2.parquet', 'file3.parquet']);
    reading_list = ", ".join([f"'{str(f)}'" for f in annot_file_chunk])
    logging.info(f"Appending {reading_list}")

    out = con.sql(
        f"""
            CREATE OR REPLACE TABLE tmp AS SELECT
                raw_file,
                scan_number,
                peptide_sequence,
                CAST (ion_type as ion_type_type) AS ion_type,
                CAST (no AS TINYINT) AS fragment_no,
                CAST (charge AS TINYINT) AS charge,
                neutral_loss,
                intensity,
                theoretical_mass as theoretical_fragment_mz,
                fragment_score,
            FROM read_parquet([{reading_list}], filename = false)
            WHERE neutral_loss = '' AND fragment_score > 0.5 AND ion_type IN ('a', 'b', 'c', 'x', 'y', 'z');

            INSERT INTO local_annotations
            SELECT
                filtered_meta.raw_file as raw_file,
                filtered_meta.scan_number as scan_number,
                peptide_sequence,
                CAST (precursor_charge AS TINYINT) AS precursor_charge,
                CAST (ion_type as ion_type_type) AS ion_type,
                fragment_no,
                neutral_loss,
                CAST (charge AS TINYINT) AS charge,
                intensity,
                -- mz, -- This is the observd mz
                theoretical_fragment_mz,
                precursor_mz, -- "same" as mz
                CAST (fragmentation AS fragmentation_type) AS fragmentation,
                CAST (mass_analyzer AS mass_analyzer_type) AS mass_analyzer,
                retention_time,
                indexed_retention_time,
                orig_collision_energy,
                aligned_collision_energy
            FROM tmp
            INNER JOIN filtered_meta
            ON tmp.peptide_sequence = filtered_meta.modified_sequence
            AND tmp.scan_number = filtered_meta.scan_number
            AND tmp.raw_file = filtered_meta.raw_file
            -- ORDER BY peptide_sequence DESC; -- 3 min with sorting, 11s without
        """
    )

logging.info("Created local annotation table")
count = con.sql("SELECT COUNT(*) FROM local_annotations").fetchone()[0]
logging.info(f"Number of annotations: {count}")
glimpse = con.sql("SELECT * FROM local_annotations LIMIT 5").to_df()
if is_in_notebook():
    print(glimpse)

# Create indices ...
# logging.info("Creating indices")
# con.sql(
#     "CREATE INDEX idx_local_annotations_peptide_sequence ON local_annotations(peptide_sequence)"
# )
# logging.info("Created indices")


# This ion has A LOT of annotated neutral losses, just making sure
# They have been filtered out (length should be 1)

scannr = "45741"
file_prefix = "01829a_BA3-TUM_third_pool_3_01_01-2xIT"
frag_no = 18
frag_ion_type = "y"
frag_charge = 2

query = f"""
SELECT
    *
FROM local_annotations
WHERE raw_file LIKE '{file_prefix}%'
AND scan_number = {scannr}
AND fragment_no = {frag_no}
AND ion_type = '{frag_ion_type}'
AND charge = {frag_charge}
ORDER BY peptide_sequence, fragment_no
"""

print(con.sql(query).to_df())


out1 = con.sql(
    """
    EXPLAIN ANALYZE
    CREATE OR REPLACE TABLE middle_tmp AS
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
            local_annotations
    ) AS inner_tmp
    GROUP BY peptide_sequence, mass_analyzer, fragmentation, orig_collision_energy
    """
)

out2 = con.sql(
    """
    EXPLAIN ANALYZE
    CREATE OR REPLACE TABLE joint_local_annotations AS
    SELECT
        middle_tmp.peptide_sequence AS peptide_sequence, 
        middle_tmp.mass_analyzer AS mass_analyzer,
        middle_tmp.fragmentation AS fragmentation,
        middle_tmp.orig_collision_energy AS orig_collision_energy,
        num_peptide_spec,                             
        raw_file,                             
        scan_number,                             
        precursor_charge,                             
        ion_type,                             
        fragment_no,                             
        charge,                             
        intensity,                             
        theoretical_fragment_mz,                             
        precursor_mz,                             
        retention_time,                             
        indexed_retention_time,                             
        aligned_collision_energy,
    FROM middle_tmp
    INNER JOIN local_annotations
    ON middle_tmp.peptide_sequence = local_annotations.peptide_sequence
    AND middle_tmp.mass_analyzer = local_annotations.mass_analyzer
    AND middle_tmp.fragmentation = local_annotations.fragmentation
    AND middle_tmp.orig_collision_energy = local_annotations.orig_collision_energy
    """
)

print(out1["explain_value"].to_df().iloc[0, 0])
print(out2["explain_value"].to_df().iloc[0, 0])

count = con.sql("SELECT COUNT(*) FROM joint_local_annotations").fetchone()[0]
glimpse = con.sql("SELECT * FROM joint_local_annotations LIMIT 5").to_df()

logging.info(f"Number of annotations: {count}")
if is_in_notebook():
    print(glimpse)


logging.info("Dropping local annotations")
con.sql("DROP TABLE local_annotations")
logging.info("Dropped local annotations")


first_pass = True
for i in range(1, 35):
    print(f"Processing fragment {i}")
    out3 = con.sql(
        f"""
        {'EXPLAIN ANALYZE' if first_pass else ''}
        {'CREATE OR REPLACE TABLE averaged_annotations AS' if first_pass else 'INSERT INTO averaged_annotations'}
        SELECT
            LIST(raw_file),
            LIST(scan_number),
            peptide_sequence AS peptide_sequence,
            precursor_charge,
            ion_type AS fragment_ion_type,
            fragment_no,
            charge AS fragment_charge,
            fragmentation,
            mass_analyzer,
            orig_collision_energy,
            theoretical_fragment_mz,
            num_peptide_spec,                                -- this is the number of spectra that were averaged
            SUM(intensity) AS total_intensity,
            SUM(intensity)/MAX(num_peptide_spec)
                AS fragment_intensity,
            MAX(intensity)
                AS max_fragment_intensity_prior_to_averaging,
            MEAN(precursor_mz)
                AS precursor_mz,
            MEAN(retention_time)
                AS retention_time,
            MEAN(indexed_retention_time)
                AS indexed_retention_time,
            MEAN(aligned_collision_energy)
                AS aligned_collision_energy,
            COUNT(*)
                AS num_averaged_peaks,
            COUNT(DISTINCT raw_file || '::' || scan_number)
                AS num_spectra,
            COUNT(DISTINCT raw_file || '::' || scan_number) / MAX(num_peptide_spec) 
                AS pct_averaged -- the percentage of spectra that contained this fragment
        FROM joint_local_annotations
        WHERE fragment_no = {i}
        GROUP BY
            peptide_sequence,
            precursor_charge,
            fragment_ion_type,
            fragment_no,
            fragment_charge,
            fragmentation,
            mass_analyzer,
            orig_collision_energy,
            theoretical_fragment_mz,
            num_peptide_spec,                                -- this is the number of spectra that were averaged
        -- ORDER BY local_annotations.peptide_sequence, local_annotations.fragment_no
        """
    )
    if first_pass:
        first_pass = False
        print(out3["explain_value"].to_df().iloc[0, 0])

logging.info("Created averaged annotations table")

count = con.sql("SELECT COUNT(*) FROM averaged_annotations").fetchone()[0]
logging.info(f"Number of annotations: {count}")
glimpse = con.sql("SELECT * FROM averaged_annotations LIMIT 5").to_df()
if is_in_notebook():
    print(glimpse)


logging.info("Dropping joint_local_annotations")
con.execute("DROP TABLE joint_local_annotations")
logging.info("Dropped joint_local_annotations")


if is_in_notebook():
    print(
        con.sql(
            "SELECT * from averaged_annotations " "ORDER BY pct_averaged DESC LIMIT 10"
        ).to_df()
    )

    print(
        con.sql(
            "SELECT * from averaged_annotations " "ORDER BY pct_averaged ASC LIMIT 10"
        ).to_df()
    )


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
        c=df["num_spectra"] / df["num_peptide_spec"],
        s=(df["num_spectra"] / df["num_peptide_spec"]) * 100,
    )
    plt.ylabel("Percetage of averaged spectra")
    plt.xlabel("max intensity pre-agg")
    plt.colorbar()
    plt.title("Pct of averaged spectra vs max intensity pre-agg, top 100K")
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

# Iteratively add sequences by the combination of
# precursor_charge, fragmentation, mass_analyzer

first_round = True

precursor_charges = [3, 2, 1, 4, 5, 6, 7]
for pc in precursor_charges:
    for frag in ["HCD", "CID", "ETD"]:
        for ma in ["ITMS", "FTMS"]:
            logging.info(f"Processing {pc}, {frag}, {ma}")
            out = con.sql(
                f"""
                {'EXPLAIN ANALYZE' if first_round else ''}
                {'CREATE OR REPLACE TABLE nested_annotations AS' if first_round else 'INSERT INTO nested_annotations'}
                SELECT
                    peptide_sequence,
                    -- TODO: Add stripped peptide here
                    precursor_charge,
                    LIST (fragment_ion_type),
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
                WHERE precursor_charge = {pc}
                AND fragmentation = '{frag}'
                AND mass_analyzer = '{ma}'
                -- pretty liberal filter, I am assuming that peaks that were not averaged 
                -- are low abundance ... BUT who knows ...
                AND pct_averaged > 0.1 
                GROUP BY peptide_sequence, precursor_charge, fragmentation, mass_analyzer, orig_collision_energy
                """
            )
            if first_round:
                logging.info(out["explain_value"].to_df().iloc[0, 0])
            first_round = False

logging.info("Created nested annotations table")


con.sql("SELECT * FROM nested_annotations LIMIT 5").to_df()


logging.info("Dropping averaged_annotations")
con.execute("DROP TABLE averaged_annotations")
logging.info("Dropped averaged_annotations")


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
    logging.info(f"Writing batch {index}")
    batch.to_parquet(f"nested_annotations_{index}.parquet")
    if index == 0:
        print(batch.head(10))
    index += 1


con.sql("SHOW tables;")


# con.sql("PRAGMA storage_info('averaged_annotations');").to_df()["count"].sum() / 1e6
# # 50.8
# con.sql("PRAGMA storage_info('filtered_meta');").to_df()["count"].sum() / 1e6
# # 4.2
# con.sql("PRAGMA storage_info('joint_local_annotations');").to_df()["count"].sum() / 1e6
# # 106.75
# con.sql("PRAGMA storage_info('local_annotations');").to_df()["count"].sum() / 1e6
# # 100.4
# con.sql("PRAGMA storage_info('nested_annotations');").to_df()["count"].sum() / 1e6
# # 9.3
# con.sql("PRAGMA storage_info('rt_qc');").to_df()["count"].sum() / 1e6
# # 0.06
# con.sql("PRAGMA storage_info('middle_tmp');").to_df()["count"].sum() / 1e6
# # 0.24


# TODO add splitting
# TODO add split set
con.close()

