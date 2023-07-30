
# poetry run python -m pip install git+https://github.com/wilhelm-lab/PROSPECT
import prospectdataset as prospect 
import prospectdataset.download as pspdl

links = pspdl.unique_urls(pspdl.filter_relevant_urls(pspdl.get_all_urls(pspdl.AVAILABLE_DATASET_URLS["prospect"])))
print("Prefix: /record/6602020/files/")
for l in links:
    print(f"'{l.replace('?download=1', '').replace('/record/6602020/files/', '')}'")


print("Prefix: /record/8003138/files/")
ptm_links = pspdl.unique_urls(pspdl.filter_relevant_urls(pspdl.get_all_urls(pspdl.AVAILABLE_DATASET_URLS["tmt"])))
for l in ptm_links:
    print(f"'{l.replace('?download=1', '').replace('/record/8003138/files/', '')}'")


multi_ptm_links = pspdl.unique_urls(pspdl.filter_relevant_urls(pspdl.get_all_urls(pspdl.AVAILABLE_DATASET_URLS["multi_ptm"])))
print("Prefix: /record/7998644/files/")
for l in multi_ptm_links:
    print(f"'{l.replace('?download=1', '').replace('/record/7998644/files/', '')}'")

# This does not support resume downloads
# prospect.download_dataset(task="all", save_directory = "prospect_data/")