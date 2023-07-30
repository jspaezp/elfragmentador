import polars as pl
import itertools
import elfragmentador as ef
from elfragmentador.config import CONFIG
from pathlib import Path
from tqdm.auto import tqdm

ION_SERIES = CONFIG.ion_series
FRAGMENT_POSITIONS = CONFIG.fragment_positions

NAME_ALIASES = {
    "ion_charges": "charge",
    "fragment_positions": "no",
    "ion_series": "ion_type",
}

FRAGMENT_ORDER = []
fragment_queries = list(
    itertools.product(*[getattr(CONFIG, n) for n in CONFIG.ion_encoding_nesting])
)
# fragment_queries = [
#     {NAME_ALIASES[w]: x[i] for i, w in enumerate(CONFIG.ion_encoding_nesting)}
#     for x in fragment_queries
# ]

print("Fragment queries:")
print(fragment_queries)

print("Fragment labels:")
print(CONFIG.fragment_labels)

schema = {
    "ion_type": pl.Utf8,
    "no": pl.Int64,
    "charge": pl.Int64,
    "experimental_mass": pl.Float64,
    "theoretical_mass": pl.Float64,
    "intensity": pl.Float64,
    "neutral_loss": pl.Utf8,
    "fragment_score": pl.Int64,
    "peptide_sequence": pl.Utf8,
    "scan_number": pl.Int64,
    "raw_file": pl.Utf8,
}


def df_as_dict(df):
    my_iter = zip(
        *(
            [df[NAME_ALIASES[w]] for w in CONFIG.ion_encoding_nesting]
            + [df["intensity"]]
        )
    )

    out = {x[:-1]: x[-1] for x in my_iter}
    return out


test_df = pl.DataFrame(
    {
        "ion_type": ["y", "b"],
        "no": [1, 2],
        "charge": [1, 1],
        "experimental_mass": [1.0, 2.0],
        "theoretical_mass": [1.0, 2.3],
        "intensity": [1.0, 0.5],
        "neutral_loss": ["", ""],
        "fragment_score": [100, 100],
        "peptide_sequence": ["AAA", "AAA"],
        "scan_number": [1, 1],
        "raw_file": ["file1", "file1"],
    }
)


def test_df_as_dict():
    df = test_df.clone()
    out = df_as_dict(df)
    print("Sample dict output:")
    print(out)
    expected_out = {(1, 1, "y"): 1.0, (1, 2, "b"): 0.5}
    assert out == expected_out


def test_df_as_vector():
    df = test_df.clone()
    out = df_as_vector(df)
    print("Sample vector output:")
    print(out)
    tot = sum(out)
    assert tot >= 1.49 and tot <= 1.51


def df_as_vector(df):
    df_dict = df_as_dict(df)
    out = [df_dict.get(x, 0.0) for x in fragment_queries]
    return out


VECTOR_DF_SCHEMA = {
    "peptide_sequence": pl.Utf8,
    "scan_number": pl.Int64,
    "raw_file": pl.Utf8,
    "vector": pl.List(pl.Float64),
}


def df_to_vector_df(df):
    assert len(set(df["peptide_sequence"])) == 1
    vector = df_as_vector(df)
    out = pl.DataFrame(
        {
            "peptide_sequence": [df["peptide_sequence"][0]],
            "scan_number": [df["scan_number"][0]],
            "raw_file": [df["raw_file"][0]],
            "vector": [vector],
        },
        schema=VECTOR_DF_SCHEMA,
    )
    return out


test_df_as_dict()
test_df_as_vector()

# pl.scan_parquet("*/*_annotation.parquet")


# filepaths = Path("prospect_data/").glob("*/*_annotation.parquet")
# for f in tqdm(filepaths):

# schema = {
#     "ion_type": pl.Utf8,
#     "no": pl.Int64,
#     "charge": pl.Int64,
#     "experimental_mass": pl.Float64, # unused
#     "theoretical_mass": pl.Float64, # unused
#     "intensity": pl.Float64,
#     "neutral_loss": pl.Utf8,
#     "fragment_score": pl.Int64, # unused RN
#     "peptide_sequence": pl.Utf8,
#     "scan_number": pl.Int64,
#     "raw_file": pl.Utf8,
# }


def main(in_path, out_path):
    print(in_path)
    print(in_path.absolute())
    out_df = (
        pl.scan_parquet(str(in_path))
        .filter(pl.col("neutral_loss") == "")
        .select(
            [
                "ion_type",
                "no",
                "charge",
                "intensity",
                "peptide_sequence",
                "scan_number",
                "raw_file",
            ]
        )
        .groupby(["raw_file", "scan_number", "peptide_sequence"])
        .apply(df_to_vector_df, VECTOR_DF_SCHEMA)
        .collect()
    )

    print("Output dataframe:")
    print(out_df)

    (out_path.parent).mkdir(exist_ok=True, parents=True)

    print(f"Writing to {out_path}")
    out_df.write_parquet(out_path)

    print("Vector lengths:")
    print(set(len(y) for y in out_df["vector"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    args = parser.parse_args()
    main(
        Path(str(args.in_path).replace(" ", "")),
        Path(str(args.out_path).replace(" ", "")),
    )
