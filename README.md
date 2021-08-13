# ElFragmentador

## ElFragmentador

This repository attempts to implement a neural net that leverages the transformer architecture to predict peptide
properties (retention time and fragmentation).

![](img/schematic.png)

## Installation

Since the project is currently in development mode, the best way to install is using pip from the cloned repo

```shell
git clone https://github.com/jspaezp/elfragmentador.git
cd elfragmentador

pip install /content/elfragmentador
```

## Usage

### Training


```shell
# Be a good person and keep track of your experiments, use wandb
$ wandb login
```

```shell
elfragmentador_train \
     --run_name onecycle_5e_petite_ndl4 \
     --scheduler onecycle \
     --max_epochs 5 \
     --lr_ratio 25 \
     --terminator_patience 20 \
     --lr 0.00005 \
     --gradient_clip_val 1.0 \
     --dropout 0.1 \
     --nhead 4 \
     --nhid 512 \
     --ninp 224 \
     --num_decoder_layers 4 \
     --num_encoder_layers 2 \
     --batch_size 400 \
     --accumulate_grad_batches 1 \
     --precision 16 \
     --gpus 1 \
     --progress_bar_refresh_rate 5 \
     --data_dir  /content/20210217-traindata
```

### Prediction

#### Check performance

I have implemented a way to compare the predictions of
the model with an `.sptxt` file. I generate them by using
`comet > mokapot > spectrast` but alternatives can be used. 

```shell
elfragmentador_evaluate --sptxt {my_sptxt_file} {path_to_my_checkpoint}
```

#### Predict Spectra

You can use it from python like so ...

```python
checkpoint_path = "some/path/to/a/checkpoint"
model = PepTransformerModel.load_from_checkpoint(checkpoint_path)

# Set the model as evaluation mode
_ = model.eval()
model.predict_from_seq("MYPEPTIDEK", charge=2, nce=27.0)
```

If you want to use graphical interface, I am currently working in
a flask app to visualize the results.

It can be run using flask.

```shell
git clone https://github.com/jspaezp/elfragmentador.git
cd elfragmentador/viz_app

# Here you can install the dependencies using poetry
python main.py
```

## Why transformers?

Because we can... Just kidding

The transformer architecture provides several benefits over the standard approach on fragment prediction (LSTM/RNN). On the training side it allows the parallel computation of whole sequences, whilst in LSTMs one element has to be passed at a time. In addition it gives the model itself a better chance to study the direct interactions between the elements that are being passed.

On the other hand, it allows a much better interpretability of the model, since the 'self-attention' can be visualized on the input and in that way see what the model is focusing on while generating the prediction.

## Inspiration for this project

Many of the elements from this project are actually a combination of the principles shown in the [*Prosit* paper](https://www.nature.com/articles/s41592-019-0426-7) and the [Skyline poster](https://skyline.ms/_webdav/home/software/Skyline/%40files/2019-ASBMB-Rohde.pdf) on some of the elements to encode the peptides and the output fragment ions.

On the transformer side of things I must admit that many of the elements of this project are derived from [DETR:  End to end detection using transformers](https://github.com/facebookresearch/detr) in particular the trainable embeddings as an input for the decoder and some of the concepts discussed about it on [Yannic Kilcher's Youtube channel](https://youtu.be/T35ba_VXkMY) (which I highly recommend).

## Why the name?

Two main reasons ... it translates to 'The fragmenter' in spanish and the project intends to predic framgnetations. On the other hand ... The name was free in pypi.

## Resources on transformers

- An amazing illustrated guide to understand the transformer architecture: <http://jalammar.github.io/illustrated-transformer/>
- Full implementation of a transformer in pytorch with the explanation of each part: <https://nlp.seas.harvard.edu/2018/04/03/attention.html>
- Official pytorch implementation of the transformer: <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>

## How fast is it?

You can check how fast the model is in you specific system.
Right now it tests only on CPU, message me if you need GPU inference times

```shell
poetry run pytest tests/test_model.py --benchmark-histogram 
```

Currenty the inference time in an Intel i5-7260U is ~7ms, or ~140 predictions per second.

## "Common" questions

- What scale are the retention times predicted.
  - Out of the model it uses a scaled version of the Biognosys retention time
    scale, so if using the base model, you will need to multiply by 100 and then
    you will get something compatible with the iRT kit.
- Is it any good?
  - Well ... yes but if you want to see if it is good for you own data I have
    added an API to test the model on a spectral library (made with spectrast).
    Just get a checkpoint of the model,
    run the command: `elfragmentador_evaluate {your_checkpoint.ckpt} {your_splib.sptxt}`
  - TODO add some benchmarking metrics to this readme ...
- Crosslinked peptides?
  - No
- ETD ?
  - No
- CID ?
  - No
- Glycosilation ?
  - No
- Negative Mode ?
  - No
- No ?
  - Not really ... I think all of those are interesting questions but
    AS IT IS RIGHT NOW it is not within the scope of the project. If you want
    to discuss it, write an issue in the repo and we can see if it is feasible.

### Known Issues

- When setting `--max_spec` on `elfragmentador_evaluate --sptxt`, the retention time accuracy is not calculated correctly because the retention times are scaled within the selected range. Since the spectra are subset in their import order, therefore only the first-eluting peptides are used.

### TODO list

#### Urgent

- Decouple to a different package with less dependencies
  the inference side of things
- Make a better logging output for the training script
- Complete dosctrings and add documentation website
- Allow training with missing values (done for RT, not for spectra)
- Migrate training data preparation script to snakemake
  - In Progress

#### Possible

- Add neutral losses specific to some PTMs
- consider if using pyteomics as  a backend for most ms-related tasks
- Translate annotation functions (getting ions) to numpy/torch
- Add weights during training so psms that are more likel to be false positives weight less

#### If I get time

- Write ablation models and benchmark them (remove parts of the model and see how much worse it gets without it)

## Acknowledgements

1. Purdue Univ for the computational resources for the preparation of the data (Brown Cluster).
2. Pytorch Lightning Team ... without this being open sourced this would not be posible.
3. Weights and Biases (same as above).
