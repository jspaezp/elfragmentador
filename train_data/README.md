
# Preparing the data for training ElFragmentador

## Downloading the data

The data uset to train elfragmentador has two main sources:
1. RT and Intensity data is acquired from the prospect zenodo repository (Mathias Wilhelm's lab, https://zenodo.org/record/6602020).
2. Ion mobility and tuning data for tims is acquired from pride (Florian Meijer's paper).

Since the download speeds and sizes are quite large, I am generating a mirror of the data in google cloud storage.

What I had to do in the cloud was to run the `prospect_data/get.zsh` script
in a cloud instance. This took ~ 12 hours total (but this means that when I want to use it again, I can download it).
