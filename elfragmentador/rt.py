from pathlib import Path
from typing import Union, List

from tqdm.auto import tqdm
import pandas as pd
from elfragmentador import constants
from elfragmentador.evaluate import polyfit


def calculate_file_iRT(file: Union[Path, str]) -> pd.DataFrame:
    df = pd.read_csv(str(file))
    df["RT"] = df["Min Start Time"] + df["Max End Time"]
    fits = {}
    for g, sub_df in df.groupby("File Name"):
        irt_sub_df = sub_df[
            [x in constants.IRT_PEPTIDES for x in sub_df["Peptide Modified Sequence"]]
        ].copy()
        if len(irt_sub_df) < 4:
            continue

        irt_sub_df["iRT"] = [
            constants.IRT_PEPTIDES[x]["irt"]
            for x in irt_sub_df["Peptide Modified Sequence"]
        ]
        fit = polyfit(irt_sub_df["RT"], irt_sub_df["iRT"])
        fits.update({g: fit})

    pred_irt = (
        lambda rt, poly: None
        if poly is None
        else rt * poly["polynomial"][0] + poly["polynomial"][1]
    )
    df["Calculated iRT"] = [
        pred_irt(y, fits.get(x, None)) for x, y in zip(df["File Name"], df["RT"])
    ]
    return df.dropna().copy().reindex()


def calculate_multifile_iRT(filelist: List[Union[str, Path]]):
    out_dfs = (calculate_file_iRT(x) for x in tqdm(filelist))

    out_df = pd.concat(out_dfs)
    group_cols = [x for x in list(out_df) if "Sequence" in x]
    gdf = (
        out_df.groupby(group_cols)
        .aggregate({"Calculated iRT": ["mean", "std", "count"]})
        .fillna(0)
    )
    gdf.columns = [" ".join(col) for col in gdf.columns.values]
    gdf.sort_values("Calculated iRT std")
    return gdf
