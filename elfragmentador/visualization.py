import logging

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import networkx as nx
import numpy as np
import pandas as pd
import torch

from elfragmentador import annotate, constants, encoding_decoding
from elfragmentador.model import PepTransformerModel


class SelfAttentionExplorer(torch.no_grad):
    """
    SelfAttentionExplorer lets you explore self-attention with a context
    manager.

    It is a context manager that takes a PepTransformerModel and wraps the transformer
    layers to save the self-attention matrices during its activity. Once it closes, the
    hooks are removed but the attention matrices are kept.

    Later these matrices can be explored. Check the examples for how to get them.

    Examples
    --------

    >>> model = PepTransformerModel() # Or load the model from a checkpoint
    >>> _ = model.eval()
    >>> with SelfAttentionExplorer(model) as sea:
    ...     _ = model.predict_from_seq(seq="MYPEPTIDEK",charge= 2, nce=30)
    ...     _ = model.predict_from_seq(seq="MY[PHOSPHO]PEPTIDEK", charge=2, nce=30)
    >>> out = sea.get_encoder_attn(layer=0, index=0)
    >>> type(out)
    <class 'pandas.core.frame.DataFrame'>
    >>> list(out)
    ['n1', 'M2', 'Y3', 'P4', 'E5', 'P6', 'T7', 'I8', 'D9', 'E10', 'K11', 'c12']
    >>> out = sea.get_decoder_attn(layer=0, index=0)
    >>> type(out)
    <class 'pandas.core.frame.DataFrame'>
    >>> list(out)[:5]
    ['z1b1', 'z1b2', 'z1b3', 'z1b4', 'z1b5']
    """

    def __init__(self, model: PepTransformerModel):
        logging.info("Initializing SelfAttentionExplorer")
        super().__init__()

        self.encoder_viz = {}
        self.decoder_viz = {}
        self.aa_seqs = {}
        self.charges = {}
        self.handles = []

        encoder = model.encoder.transformer_encoder
        decoder = model.decoder.trans_decoder

        encoder_hook = self._make_hook_transformer_layer(self.encoder_viz)
        for layer in range(0, len(encoder.layers)):
            logging.info(f"Adding hook to encoder layer: {layer}")
            handle = encoder.layers[layer].self_attn.register_forward_hook(encoder_hook)
            self.handles.append(handle)

        decoder_hook = self._make_hook_transformer_layer(self.decoder_viz)
        for layer in range(0, len(decoder.layers)):
            logging.info(f"Adding hook to decoder layer: {layer}")
            handle = decoder.layers[layer].self_attn.register_forward_hook(decoder_hook)
            self.handles.append(handle)

        aa_hook = self._make_hook_aa_layer(self.aa_seqs)
        handle = model.encoder.aa_encoder.aa_encoder.register_forward_hook(aa_hook)
        self.handles.append(handle)

        charge_hook = self._make_hook_charge(self.charges)
        handle = model.decoder.charge_encoder.register_forward_hook(charge_hook)
        self.handles.append(handle)

    def __enter__(self):
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        logging.info("Removing Handles")
        super().__exit__(exc_type, exc_value, exc_traceback)
        for h in self.handles:
            h.remove()

        # TODO consider if all self attention matrices/dataframes should
        #      be calculated on exit. Or even the bipartite graphs

    def __repr__(self):
        out = (
            ">>> SelfAttentionExplorer <<<<\n\n"
            ">> AA sequences (aa_seqs):\n"
            f"{self.aa_seqs.__repr__()}\n"
            ">> Charges (charges):\n"
            f"{self.charges.__repr__()}\n"
            ">> Encoder vizs (encoder_viz)"
            f" {list(self.encoder_viz.values())[0].shape}:\n"
            f"{self.encoder_viz.__repr__()}\n"
            ">> Decoder vizs (decoder_viz)"
            f" {list(self.decoder_viz.values())[0].shape}:\n"
            f"{self.decoder_viz.__repr__()}"
        )

        return out

    @staticmethod
    def _make_hook_transformer_layer(target):
        def hook_fn(m, i, o):
            # The output of the self attention layer is the value and the weights
            # of the self attention
            self_attention_weights = m.forward(*i)[1]
            if target.get(m, None) is None:
                target[m] = self_attention_weights
            else:
                target[m] = torch.cat([target[m], self_attention_weights])

        return hook_fn

    @staticmethod
    def _make_hook_aa_layer(target):
        def hook_fn(m, i, o):
            if target.get(m, None) is None:
                target[m] = []
            target[m].append(
                encoding_decoding.decode_mod_seq(
                    [int(x) for x in i[0]], clip_explicit_term=False
                )
            )

        return hook_fn

    @staticmethod
    def _make_hook_charge(target):
        def hook_fn(m, i, o):
            if target.get(m, None) is None:
                target[m] = []
            target[m].extend([int(x) for x in i[1]])

        return hook_fn

    @staticmethod
    def _norm(attn):
        attn = (attn - np.mean(attn)) / np.std(attn)
        return attn

    def get_encoder_attn(self, layer: int, index: int = 0, norm=False) -> pd.DataFrame:
        seq = list(self.aa_seqs.values())[0][index]
        attn = list(self.encoder_viz.values())[layer][index][: len(seq), : len(seq)]
        if norm:
            attn = self.norm(attn)

        names = [x + str(i + 1) for i, x in enumerate(seq)]
        attn = pd.DataFrame(attn.clone().detach().numpy(), index=names, columns=names)
        return attn[::-1]

    def get_decoder_attn(self, layer: int, index: int = 0, norm=True) -> pd.DataFrame:
        attn = list(self.decoder_viz.values())[layer][index].clone().detach().numpy()
        seq = list(self.aa_seqs.values())[0][index]
        charge = list(self.charges.values())[0][index]
        theo_ions = list(annotate.get_peptide_ions(seq).keys())
        theo_ions = [x for x in theo_ions if int(x[1]) <= charge]
        if norm:
            attn = self._norm(attn)

        names = constants.FRAG_EMBEDING_LABELS
        attn = pd.DataFrame(attn, index=names, columns=names)

        # TODO add charge filtering
        return attn[theo_ions].loc[theo_ions][::-1]


def make_bipartite(x):
    """
    Makes a bipartite graph from a data frame whose col and row indices are the
    same.
    """

    B = nx.Graph()
    B.add_nodes_from(
        [x for x in x.index], bipartite=0
    )  # Add the node attribute "bipartite"
    B.add_nodes_from([x + "_" for x in x.index], bipartite=1)

    for index1 in x.index:
        for index2 in x.index:
            B.add_edges_from(
                [
                    (index1, index2 + "_"),
                ],
                weight=x[index1].loc[index2],
            )

    return B


def plot_bipartite_seq(B):
    """
    Plots a bipartite graph from a sequence self-attention.

    expects names to be in the form of X[index] and X[index]_
    """
    if plt is None:
        raise ImportError(
            "Matplotlib is not installed, please install and re-load elfragmentador"
        )

    # Separate by group
    l, r = nx.bipartite.sets(B)
    pos = {}
    # Update position for node from each group
    pos.update((node, (int(node[1:]), 1)) for node in l)
    pos.update((node, (int(node[1:-1]), 2)) for node in r)
    weights = list(nx.get_edge_attributes(B, "weight").values())
    mean_weight = np.array(weights).mean()
    min_weight = np.array(weights).min()
    weights = [x if x > mean_weight else min_weight for x in weights]

    nx.draw(
        B,
        pos=pos,
        edge_color=weights,
        edge_cmap=plt.cm.Blues,
        with_labels=True,
        width=3,
        alpha=0.5,
    )
    plt.show()
