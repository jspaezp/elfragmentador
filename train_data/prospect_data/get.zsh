#!/usr/bin/zsh

prospect_files=(
    'README.md'
    'Thermo_SRM_Pool.zip'
    'Thermo_SRM_Pool_meta_data.parquet'
    'TUM_aspn.zip'
    'TUM_aspn_meta_data.parquet'
    'TUM_first_pool_1.zip'
    'TUM_first_pool_2.zip'
    'TUM_first_pool_3.zip'
    'TUM_first_pool_meta_data.parquet'
    'TUM_HLA.zip'
    'TUM_HLA2_1.zip'
    'TUM_HLA2_2.zip'
    'TUM_HLA2_meta_data.parquet'
    'TUM_HLA_meta_data.parquet'
    'TUM_isoform_1.zip'
    'TUM_isoform_2.zip'
    'TUM_isoform_meta_data.parquet'
    'TUM_lysn.zip'
    'TUM_lysn_meta_data.parquet'
    'TUM_missing_first.zip'
    'TUM_missing_first_meta_data.parquet'
    'TUM_proteo_TMT.zip'
    'TUM_proteo_TMT_meta_data.parquet'
    'TUM_second_addon.zip'
    'TUM_second_addon_meta_data.parquet'
    'TUM_second_pool_1.zip'
    'TUM_second_pool_2.zip'
    'TUM_second_pool_3.zip'
    'TUM_second_pool_meta_data.parquet'
    'TUM_third_pool.zip'
    'TUM_third_pool_meta_data.parquet'
)

tmt_files=(
    'TMT_Thermo_SRM_Pool.zip'
    'TMT_Thermo_SRM_Pool_meta_data.parquet'
    'TMT_TUM_aspn.zip'
    'TMT_TUM_aspn_meta_data.parquet'
    'TMT_TUM_first_pool.zip'
    'TMT_TUM_first_pool_meta_data.parquet'
    'TMT_TUM_HLA.zip'
    'TMT_TUM_HLA_meta_data.parquet'
    'TMT_TUM_isoform.zip'
    'TMT_TUM_isoform_meta_data.parquet'
    'TMT_TUM_lysn.zip'
    'TMT_TUM_lysn_meta_data.parquet'
    'TMT_TUM_missing_first.zip'
    'TMT_TUM_missing_first_meta_data.parquet'
    'TMT_TUM_proteo_TMT.zip'
    'TMT_TUM_proteo_TMT_meta_data.parquet'
    'TMT_TUM_second_addon.zip'
    'TMT_TUM_second_addon_meta_data.parquet'
    'TMT_TUM_second_pool.zip'
    'TMT_TUM_second_pool_meta_data.parquet'
    'TMT_TUM_third_pool.zip'
    'TMT_TUM_third_pool_meta_data.parquet'
)

ptm_files=(
    'TUM_mod_acetylated.zip'
    'TUM_mod_acetylated_meta_data.parquet'
    'TUM_mod_citrullination_l.zip'
    'TUM_mod_citrullination_l_meta_data.parquet'
    'TUM_mod_imp_pSTY.zip'
    'TUM_mod_imp_pSTY_meta_data.parquet'
    'TUM_mod_monomethyl.zip'
    'TUM_mod_monomethyl_meta_data.parquet'
    'TUM_mod_OGalNAc.zip'
    'TUM_mod_OGalNAc_meta_data.parquet'
    'TUM_mod_OGlcNAc.zip'
    'TUM_mod_OGlcNAc_meta_data.parquet'
    'TUM_mod_pS.zip'
    'TUM_mod_pS_meta_data.parquet'
    'TUM_mod_pT.zip'
    'TUM_mod_pT_meta_data.parquet'
    'TUM_mod_pY.zip'
    'TUM_mod_pY_meta_data.parquet'
    'TUM_mod_pyroGlu.zip'
    'TUM_mod_pyroGlu_meta_data.parquet'
    'TUM_mod_ubi.zip'
    'TUM_mod_ubi_meta_data.parquet'
    'TUM_nterm_ac.zip'
    'TUM_nterm_ac_meta_data.parquet'
    'TUM_perm_pS.zip'
    'TUM_perm_pS_meta_data.parquet'
    'TUM_perm_pT.zip'
    'TUM_perm_pT_meta_data.parquet'
    'TUM_perm_pY.zip'
    'TUM_perm_pY_meta_data.parquet'
)

for target_file in $prospect_files ; do
    echo $target_file
    wget -c https://zenodo.org/record/6602020/files/${target_file}\?download\=1  -O ${target_file}
done

for target_file in $tmt_files ; do
    echo $target_file
    wget -c https://zenodo.org/record/8003138/files/${target_file}\?download\=1  -O ${target_file}
done

for target_file in $ptm_files ; do
    echo $target_file
    wget -c https://zenodo.org/record/7998644/files/${target_file}\?download\=1  -O ${target_file}
done

for f in *.zip ; do
    7za x ${f}
done 

gsutil -m cp -n *.parquet */*.parquet gs://jspp_prospect_mirror

# Bigquery was very expensive for this data layout.
# bq load --source_format=PARQUET \
#     --replace --clustering_fields="modified_sequence,raw_file" \
#     prospect_proteomics.metadata \
#     "gs://jspp_prospect_mirror/*_meta_data.parquet"
# 
# bq load --source_format=PARQUET \
#     --replace --clustering_fields="modified_sequence,raw_file" \
#     prospect_proteomics.annotations \
#     "gs://jspp_prospect_mirror/*_annotations.parquet"
