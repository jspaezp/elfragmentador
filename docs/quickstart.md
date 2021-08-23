
# Quickstart

## Installation

User Install:

```shell
pip install elfragmentador
```

Development install:

```shell
git clone https://github.com/jspaezp/elfragmentador.git
cd elfragmentador

pip install /content/elfragmentador
# or ...
poetry install
```

## Usage

### Prediction

I have implemented various ways to get predictions from the models.

If you want to generate an in-silico spectral library, you can use
a csv file with the columns 'Modified Sequence', 'CE' (collision energy),
and 'Precursor Charge'.

The output file is a `.sptxt` file, as specified by spectrast.

```shell
elfragmentador_predict_csv --csv {input csv file} --out {output .sptxt file}
```

If you want to append the cosine similarity with predictions and
error with respect to a predicted retention time to a percolator input file (.pin) file:

```
elfragmentador_append_pin --pin {input .pin} --out {output .pin}
```

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
import elfragmentador as ef
from elfragmentador.model import PepTransformerModel

checkpoint_path = "some/path/to/a/checkpoint"
#or
checkpoint_path = ef.DEFAULT_CHECKPOINT
model = PepTransformerModel.load_from_checkpoint(checkpoint_path)

# Set the model as evaluation mode
_ = model.eval()
tensor_predictions = model.predict_from_seq("MYPEPTIDEK", charge=3, nce=27.0)
# PredictionResults(irt=tensor([0.2022], grad_fn=<SqueezeBackward1>), spectra=tensor([0.0000e+00, ...grad_fn=<SqueezeBackward1>))

#or ...
import matplotlib.pyplot as plt
spectrum_prediction = model.predict_from_seq("MYPEPTIDEK", charge=3, nce=27.0, as_spectrum=True)
spectrum_prediction.plot(ax = ax)
plt.show()

```

![](img/spectrum.png)

```python
print(tensor_predictions.to_sptxt())
# Name: MYPEPTIDEK/3
# MW: 1221.5587417
# PrecursorMZ: 408.19352370033334
# FullName: MYPEPTIDEK/3 (HCD)
# Comment: CollisionEnergy=27.0 Origin=ElFragmentador_v0.48.2 RetentionTime=1213.3010923862457 iRT=20.221684873104095
# Num Peaks: 19
# 132.047761467   0.004137382842600346    "?"
# 147.112804167   0.5590275526046753      "?"
# 295.111090467   0.449807733297348       "?"
# 276.15539716700005      1.0     "?"
# 392.16385446699996      0.033638376742601395    "?"
# 391.18234016700006      0.6440941095352173      "?"
# 521.206447467   0.1925528198480606      "?"
# 504.2664041670001       0.5312701463699341      "?"
# 618.259211467   0.014472639188170433    "?"
# 605.314083167   0.2590445280075073      "?"
# 719.306890467   0.03478344529867172     "?"
# 702.3668471670001       0.8792285919189453      "?"
# 831.4094401670001       0.08646849542856216     "?"
# 196.09480831700003      0.007271825801581144    "?"
# 252.63684031700004      0.0016646143049001694   "?"
# 309.633243967   0.01808040589094162     "?"
# 360.157083467   0.010775345377624035    "?"
# 351.68706181700003      0.37845638394355774     "?"
# 464.73474031700005      0.14382316172122955     "?"
```

If you want to use graphical interface, I am currently working in
a flask app to visualize the results.

It can be run using flask.

```shell
git clone https://github.com/jspaezp/elfragmentador.git
cd elfragmentador/viz_app

# Here you can install the dependencies using poetry
python main.py

# and then go to http://localhost:5000/
# in your browser
```

### Training

Training is handled by calling a training script from the shell... this would be an example.

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