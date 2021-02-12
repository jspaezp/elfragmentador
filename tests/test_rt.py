from elfragmentador import rt


def test_calculating_rts_doesnt_fail(shared_datadir):
    file = shared_datadir / "./combined_pl_TUM_proteo_TMT_3_skyline.rt.csv"
    rt.calculate_file_iRT(file)
    rt.calculate_multifile_iRT([file, file])
