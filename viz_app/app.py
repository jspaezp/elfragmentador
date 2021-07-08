import logging
import os
import torch
from elfragmentador.model import PepTransformerModel
from flask import Flask, render_template

import base64
from io import BytesIO

from matplotlib.figure import Figure
from matplotlib import pyplot as plt

import logging

logging.basicConfig(
    filename="app.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(name)s - %(levelname)s - %(message)s",
)


def make_fig():
    # Generate the figure **without using pyplot**.
    fig = Figure(figsize=(15, 6))
    ax = fig.subplots()
    ax.plot([1, 2])

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data


def make_spec_fig(spectrum):
    # Generate the figure **without using pyplot**.
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    spectrum.plot(ax=ax)

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data


# export FLASK_CHECKPOINT=blablabla.ckpt export FLASK_APP=app.py ; export FLASK_ENV=development ; flask run

app = Flask(__name__)
model = PepTransformerModel()
model.eval()


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/spec/<peptide>/<z>/<nce>")
def showpeptide(peptide, z=2, nce=27.0):
    logging.info("Starting prediction for peptide {peptide}")
    pred = model.predict_from_seq(
        peptide, charge=int(z), nce=float(nce), as_spectrum=True
    )
    print(pred)
    fig = make_spec_fig(pred)
    rendered_template = render_template(
        "viz_template.html",
        peptide=peptide,
        img_data=fig,
        irt=str(pred.rt),
        sptxt=pred.to_sptxt(),
    )

    return rendered_template


if __name__ == "__main__":
    pass
