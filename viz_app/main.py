import logging
import os
import torch
import elfragmentador as ef
from elfragmentador.model import PepTransformerModel
from flask import Flask, render_template, request, url_for, redirect


import base64
from io import BytesIO

from matplotlib.figure import Figure
from matplotlib import pyplot as plt

import logging

logging.basicConfig(
    filename="app.log",
    filemode="w",
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)


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
model = PepTransformerModel.load_from_checkpoint(ef.DEFAULT_CHECKPOINT)
model.eval()


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/handle_form", methods=["POST", "GET"])
def handle_form():
    print(request.method)
    logging.info(request.method)
    if request.method == "POST":
        print(request.form)
        sequence = request.form["sequence"]
        charge = request.form["charge"]
        nce = request.form["nce"]
        out_url = url_for("showpeptide", peptide=sequence, z=charge, nce=nce)
        return redirect(out_url)
    else:
        print(request.args)
        sequence = request.args.get("sequence")
        charge = request.args.get("charge")
        nce = request.args.get("nce")
        out_url = url_for("showpeptide", peptide=sequence, z=charge, nce=nce)
        return redirect(out_url)


@app.route("/spec/<peptide>/<z>/<nce>")
def showpeptide(peptide, z=2, nce=27.0):
    logging.info("Starting prediction for peptide {peptide}")

    with torch.no_grad():
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
    app.run(debug=True)
