import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader

from elfragmentador.datasets.dataset import ComparissonDataset
from elfragmentador.datasets.spectronaut_dataset import SpectronautLibrary
from elfragmentador.metrics import MetricCalculator


def test_spectronaut_library_loads(shared_datadir):
    path = shared_datadir / "spectronaut_csv/small_prositLib.csv"
    lib = SpectronautLibrary(path, nce=25)
    elem = lib[0]

    assert isinstance(elem, tuple)
    assert hasattr(elem, "_fields")


def test_spectronaut_library_can_be_matched(shared_datadir):
    path = shared_datadir / "spectronaut_csv/small_prositLib.csv"
    lib = SpectronautLibrary(path, nce=25)
    lib2 = SpectronautLibrary(path, nce=25)

    cds = ComparissonDataset(lib, lib2)
    cds._match_datasets()

    calculator = MetricCalculator()

    for x in DataLoader(cds, batch_size=5):
        outs = calculator(x["gt"], x["pred"])

        print(outs)
        print(x["gt"].spectra.shape)
        print(x["pred"].spectra.shape)

        for i, metric in enumerate(outs):
            for gt, pred, o in zip(x["gt"].spectra, x["pred"].spectra, metric):
                assert o < 1e-3, f"\nx={gt}\ny={pred}\no={o}, i={i}"
                assert o > -1e-3, f"\nx={gt}\ny={pred}\no={o} i={i}"


def test_spectronaut_library_can_be_compared(shared_datadir, tmpdir):
    path = shared_datadir / "spectronaut_csv/small_prositLib.csv"
    lib = SpectronautLibrary(path, nce=25)
    lib2 = SpectronautLibrary(path, nce=25)

    cds = ComparissonDataset(lib, lib2)
    cds._match_datasets()
    out = cds.compare()

    assert len(out) == 4
    assert isinstance(out, tuple)
    assert hasattr(out, "_fields")

    out_df = cds.save_data(prefix=tmpdir)
    assert isinstance(out_df, DataFrame)
    assert len(out_df) == len(out[0])

    assert (tmpdir + ".csv").exists()
    assert (tmpdir + ".feather").exists()

    # This entails it can be read
    pd.read_csv(str(tmpdir) + ".csv")
    pd.read_feather(str(tmpdir) + ".feather")
