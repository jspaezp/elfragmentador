import os
from elfragmentador.datasets.sequence_dataset import FastaDataset


def test_fasta_dataset_works(shared_datadir):
    fasta_file = (
        shared_datadir / "fasta/uniprot-proteome_UP000464024_reviewed_yes.fasta"
    )
    enzyme = "trypsin"
    missed_cleavages = 2
    min_length = 5
    charges = [2, 3]
    nces = [27, 30]

    my_ds = FastaDataset(
        fasta_file=fasta_file,
        enzyme=enzyme,
        missed_cleavages=missed_cleavages,
        min_length=min_length,
        collision_energies=nces,
        charges=charges,
    )

    my_ds[0]
    assert isinstance(my_ds[0], tuple)
    assert my_ds[0]._fields == ("seq", "mods", "charge", "nce")
