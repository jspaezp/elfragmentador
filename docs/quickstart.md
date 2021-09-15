
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
a csv file with the columns 'Modified Sequence'/'modified_sequence', 'CE'/'collision_energy' (collision energy),
and 'Precursor Charge'/'precursor_charge'.

The output file is a `.sptxt` file, as specified by spectrast.

```shell
elfragmentador_predict_csv --csv {input csv file} --out {output .sptxt file}
```

For instance, if I have a file called `peptidelist.csv` with this contents:

```txt
modified_sequence,collision_energy,precursor_charge
MMPAAALIM(ox)R,35,3
MLAPPPIM(ox)K,30,2
MRALLLIPPPPM(ox)R,30,6
```

And I run....

```shell
elfragmentador_predict_csv --csv peptidelist.csv --out lib.sptxt
```

I would get a `lib.sptxt` file that starts like this

```txt
Name: MMPAAALIM[16]R/3
MW: 1119.5602747
PrecursorMZ: 374.19403470033336
FullName: MMPAAALIM[16]R/3 (HCD)
Comment: CollisionEnergy=35.0 Origin=ElFragmentador_v0.50.0b22 RetentionTime=406044.61669921875 iRT=6767.4102783203125
Num Peaks: 17
175.118952167	1.0	"?"
263.088246467	0.01451836433261633	"?"
322.15434716699997	0.28581002354621887	"?"
360.141010467	0.0387122668325901	"?"
```

If you want to append the cosine similarity with predictions and
error with respect to a predicted retention time to a percolator input file (.pin) file:

### Rescoring

```
elfragmentador_append_pin --pin {input .pin} --out {output .pin}
```

### Check performance

I have implemented a way to compare the predictions of
the model with an `.sptxt` file. I generate them by using
`comet > mokapot > spectrast` but alternatives can be used. 

```shell
elfragmentador_evaluate --sptxt {my_sptxt_file} {path_to_my_checkpoint}
```

You will get something like this .... (plus some logging information)

```
┌────────────────────────────────────────────────────────────┐
│                             │     ▝                        │ 0.86
│                             ▗           ▝                  │ 
│                             │                              │ 
│                             │                              │ 
│                       ▗     │                              │ 0.81
│                             │                 ▝            │ 
│                             │                       ▝      │ 
│                             │                              │ 
│                 ▗           │                              │ 0.76
│                             │                              │ 
│                             │                              │ 
│           ▝                 │                              │ 
│                             │                              │ 0.71
│     ▝                       │                              │ 
│                             │                             ▝│ 
│                             │                              │ 
│▖                            │                              │ 0.66
└────────────────────────────────────────────────────────────┘
 -10.000                                                 10.002

Predicting: 100%|██████████| 2758/2758 [23:52<00:00,  1.93it/s]
Testing: 100%|█████████▉| 2749/2758 [00:06<00:00, 462.23it/s]    Scaled ground truth (y) vs scaled prediction(x) of RT
┌────────────────────────────────────────────────────────────┐
│                 ▖▖▖▖▙▙▖▄▌▄█▄▖▄▙▄▄▗▄▖▄▄▖▟▄▄ ▖▄▄ ▄▄▄▄▄▄▄▄▄▄▄▄│ 2.104
│       ▘      ▐   ▌▌▌█▙▌▐▌██▐▘▟█▖▗▚▐██▟▙████████████▛▛▀▛    │ 
│       ▖         ▌▌▌ ▙█ ▐▌███ ██▛▐▙▟██████████▛█ █▀    ▘    │ 
│       ▘    ▘▐  ▘▙▌▙ ██▖▐▌███▘▜█▜█▟███████████ ▌ ▐          │ 
│       ▘     ▗  ▖▛▌█▖██▌▐▌███▙█████████████▝▙▝              │ 1.051
│       ▘    ▖▝▗ ▘▖▙▌ █▙▌▜▛▟████████████▌██▌▗▛ ▌▗            │ 
│       ▌    ▌▟   ▌▙▌▖██▌████████████▛▛▚▞▜▟█▐▘  ▐            │ 
│       ▌    ▌▗▐  ▌▙█▞█▙▟█████████████▌▜▝▞▚▌▝   ▐ ▐          │ 
│───────▌───▝▌▐▐▖─▙██▙██████████████▜██──▐▖▙───▘▐────────────│ -0.002
│       ▖     ▟▐▟▖▙█████████████▛██▛▝▙▛   ▘▌    ▐            │ 
│       ▌▗▖▖▐▀▜█▟█████████████▌▐▛▖▀▜ ▙█      ▗ ▗             │ 
│      ▗▌  ▌▐█████████████▌█▐█▘  ▜ ▝▖▀▝ ▘▌▐▘   ▖▖            │ 
│     ▟▐▌▗███████████████▘█▐▛▘    ▗▖ ▝▗ ▗▐  ▝   ▘            │ -1.055
│    ▐▙▟█████████████▛█▜▌▄▟▙▗    ▄ ▛▖▖ ▖▝  ▟   ▐▝ ▖          │ 
│  ▖▌▟████████████▜▐▙▝██▖▖│▐▟ ▝▐ ▛  ▌▝     ▀  ▘              │ 
│▝ ▟█████████████▙▟█▌▖█▛▙█▌▗▛▘▖▝█▌▌▚▌▄ ▐ ▚▗▖                 │ 
│▙████▛██▜▜█▀▛█▘▛▝▀▀▝▝ ▜ ▘▛▝  ▝▟▘▘▘▝▘▝▘  ▝▝▘                 │ -2.108
└────────────────────────────────────────────────────────────┘
 -2                                                        3
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'median_loss_angle': 0.21370655298233032,
 'median_loss_cosine': 0.055816590785980225,
 'median_loss_irt': 255.625244140625,
 'median_scaled_se_loss': 0.011943627148866653}
--------------------------------------------------------------------------------

Testing: 100%|██████████| 2758/2758 [00:06<00:00, 450.02it/s]
       Square scaled RT error mean:0.09919782727956772
┌────────────────────────────────────────────────────────────┐
│  ▐▀▀▜                                                      │ 126,878
│  ▐ │▐                                                      │ 
│  ▐ │▐                                                      │ 
│  ▐ │▐                                                      │ 
│  ▐ │▐                                                      │ 96,119
│  ▐ │▐                                                      │ 
│  ▐ │▐                                                      │ 
│  ▐ │▐                                                      │ 
│  ▐ │▐                                                      │ 65,361
│  ▐ │▐                                                      │ 
│  ▐ │▐                                                      │ 
│  ▐ │▐                                                      │ 
│  ▐ │▐                                                      │ 34,603
│  ▐ │▐                                                      │ 
│  ▐ │▐                                                      │ 
│  ▐ │▐                                                      │ 
│▄▄▟▁│▝▀▀▜▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│ 3,844
└────────────────────────────────────────────────────────────┘
 -1                                                        14
Accumulative distribution (y) of the 1 - Square scaled RT error (x)
P90=0.115 Q3=0.039 Median=0.012 Q1=0.003 P10=0.000
┌────────────────────────────────────────────────────────────┐
│                                                       │   ▐│ 1.0
│                                                       │   ▐│ 
│                                                       │   ▐│ 
│                                                       │   ▐│ 
│                                                       │   ▐│ 0.7
│                                                       │   ▐│ 
│                                                       │   ▐│ 
│                                                       │   ▐│ 
│                                                       │   ▐│ 0.5
│                                                       │   ▐│ 
│                                                       │   ▐│ 
│                                                       │   ▐│ 
│                                                       │   ▐│ 0.3
│                                                       │   ▐│ 
│                                                       │   ▐│ 
│                                                       ▗▄▄▞▘│ 
│▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▞▀▀▘▁▁▁▁│ 0.0
└────────────────────────────────────────────────────────────┘
 -12                                                        1
      Spectra Cosine Similarity mean:0.8854953646659851
┌────────────────────────────────────────────────────────────┐
│    │                                             ▐▀▀▜      │ 57,744
│    │                                             ▐  ▐      │ 
│    │                                             ▐  ▐      │ 
│    │                                             ▐  ▐      │ 
│    │                                             ▐  ▐      │ 43,745
│    │                                             ▐  ▐      │ 
│    │                                             ▐  ▐      │ 
│    │                                             ▐  ▐      │ 
│    │                                             ▐  ▐      │ 29,746
│    │                                             ▐  ▐▄▄▄   │ 
│    │                                             ▐     ▐   │ 
│    │                                          ▐▀▀▀     ▐   │ 
│    │                                          ▐        ▐   │ 15,748
│    │                                          ▐        ▐   │ 
│    │                                       ▗▄▄▟        ▐   │ 
│    │                              ▗▄▄▄▄▄▟▀▀▀           ▐   │ 
│▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟▀▀▀▀▀▀▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▐▄▄▄│ 1,749
└────────────────────────────────────────────────────────────┘
 0                                                          1
Accumulative distribution (y) of the 1 - Spectra Cosine Similarity (x)
P90=0.986 Q3=0.974 Median=0.944 Q1=0.862 P10=0.672
┌────────────────────────────────────────────────────────────┐
│                           ▗▄▄▄▄▄▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│ 1.0
│                  ▗▄▄▄▀▀▀▀▀▘                                │ 
│              ▄▄▀▀▘                                         │ 
│         ▗▄▞▀▀                                              │ 
│       ▗▀▘                                                  │ 0.7
│      ▄▘                                                    │ 
│     ▞                                                      │ 
│    ▞                                                       │ 
│   ▞                                                        │ 0.5
│  ▗▘                                                        │ 
│  ▞                                                         │ 
│ ▗▘                                                         │ 
│ ▐                                                          │ 0.3
│ ▌                                                          │ 
│▗▘                                                          │ 
│▐                                                           │ 
│▌▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│ 0.0
└────────────────────────────────────────────────────────────┘
 0.0                                                      1.0
      Spectral Angle Similarity mean:0.7337521910667419
┌────────────────────────────────────────────────────────────┐
│    │                                          ▐▀▀▜         │ 30,182
│    │                                       ▐▀▀▀  ▐         │ 
│    │                                       ▐     ▐         │ 
│    │                                       ▐     ▐         │ 
│    │                                       ▐     ▐         │ 22,865
│    │                                       ▐     ▐         │ 
│    │                                    ▐▀▀▀     ▐         │ 
│    │                                    ▐        ▐         │ 
│    │                                    ▐        ▐         │ 15,548
│    │                                 ▗▄▄▟        ▐         │ 
│    │                                 ▐           ▐         │ 
│    │                                 ▐           ▐▄▄▄      │ 
│    │                              ▗▄▄▟              ▐      │ 8,231
│    │                        ▗▄▄▄▄▄▟                 ▐      │ 
│    │                  ▗▄▄▟▀▀▀                       ▐      │ 
│    │               ▐▀▀▀                             ▐      │ 
│▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▟▀▀▀▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▐▄▄▄▄▄▄│ 914
└────────────────────────────────────────────────────────────┘
 0                                                          1
Accumulative distribution (y) of the 1 - Spectral Angle Similarity (x)
P90=0.892 Q3=0.854 Median=0.786 Q1=0.662 P10=0.469
┌────────────────────────────────────────────────────────────┐
│                                     ▄▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│ 1.0
│                              ▄▄▄▀▀▀▀                       │ 
│                         ▗▄▞▀▀                              │ 
│                    ▗▄▄▀▀▘                                  │ 
│                  ▄▀▘                                       │ 0.7
│                ▗▀                                          │ 
│              ▗▞▘                                           │ 
│            ▗▞▘                                             │ 
│           ▗▘                                               │ 0.5
│          ▗▘                                                │ 
│         ▗▘                                                 │ 
│        ▗▘                                                  │ 
│       ▗▘                                                   │ 0.3
│      ▗▘                                                    │ 
│     ▗▘                                                     │ 
│    ▄▘                                                      │ 
│▄▄▄▀▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│ 0.0
└────────────────────────────────────────────────────────────┘
 0.0                                                      1.0

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