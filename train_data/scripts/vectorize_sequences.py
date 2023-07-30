import polars as pl
from elfragmentador.data.converter import Tensorizer
import torch

foo = pl.scan_parquet("processed_data/rt_qc.parquet")
modseqs = foo.select("modified_sequence").collect()["modified_sequence"]

with torch.no_grad():
    out = [Tensorizer.convert_string(f"{x}/2", 27.0) for x in modseqs]
    out = [(x.seq.tolist(), x.mods.tolist()) for x in out]

    out2 = pl.DataFrame(
        out,
        schema={
            "sequence_vector": pl.List(pl.List(pl.Int32)),
            "mod_vector": pl.List(pl.List(pl.Int32)),
        },
    )
    out2 = out2.with_columns(modified_sequence=modseqs)
    print(out2)

    out2.write_parquet("test.parquet")
