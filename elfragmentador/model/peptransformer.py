from __future__ import annotations

import torch.nn as nn
from loguru import logger
from ms2ml import AnnotatedPeptideSpectrum
from torch import Tensor

from elfragmentador.data.converter import DeTensorizer, Tensorizer
from elfragmentador.model.ms_transformer_layers import (
    FragmentTransformerDecoder,
    IRTDecoder,
    PeptideTransformerEncoder,
)
from elfragmentador.named_batches import PredictionResults


class PepTransformerBase(nn.Module):
    def __init__(
        self,
        num_fragments,
        num_decoder_layers: int = 6,
        num_encoder_layers: int = 6,
        nhid: int = 2024,
        d_model: int = 516,
        nhead: int = 4,
        dropout: float = 0.1,
        combine_embeds: bool = True,
        combine_encoders: bool = True,
        final_decoder="linear",
    ) -> None:
        super().__init__()
        # Peptide encoder
        self.encoder = PeptideTransformerEncoder(
            d_model=d_model,
            dropout=dropout,
            nhead=nhead,
            nhid=nhid,
            layers=num_encoder_layers,
        )

        # Peptide decoder
        self.decoder = FragmentTransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            nhid=nhid,
            layers=num_decoder_layers,
            dropout=dropout,
            num_fragments=num_fragments,
            final_decoder=final_decoder,
        )

        self.irt_decoder = IRTDecoder(
            d_model=d_model,
            dim_feedforward=nhid,
            nhead=nhead,
            n_layers=num_encoder_layers,
            dropout=dropout,
            final_decoder=final_decoder,
        )

        if combine_embeds:
            self.irt_decoder.aa_embed = self.encoder.aa_embed

        if combine_encoders:
            self.irt_decoder.encoder = self.encoder.encoder

    def forward(
        self,
        seq: Tensor,
        mods: Tensor,
        charge: Tensor,
        nce: Tensor,
    ) -> PredictionResults:
        """
        Forward Generate predictions.

        Privides the function for the forward pass to the model.

        Parameters:
            seq (Tensor): Encoded pepide sequence [B, L] (view details)
            mods (Tensor): Encoded modification sequence [B, L], by default None
            nce (Tensor): float Tensor with the charges [B, 1]
            charge (Tensor): long Tensor with the charges [B, 1], by default None

        Details:
            seq:
                The peptide is encoded as integers for the aminoacid.
                "AAA" encoded for a max length of 5 would be
                torch.Tensor([ 1,  1,  1,  0,  0]).long()
            nce:
                Normalized collision energy to use during the prediction.
            charge:
                A tensor corresponding to the charges of each of the
                peptide precursors (long)
            mods:
                Modifications encoded as integers
        """

        trans_encoder_output, mem_mask = self.encoder(seq=seq, mods=mods)

        rt_output = self.irt_decoder(seq=seq, mods=mods)

        spectra_output = self.decoder(
            memory=trans_encoder_output,
            charge=charge,
            nce=nce,
            memory_key_padding_mask=mem_mask,
        )

        return PredictionResults(irt=rt_output, spectra=spectra_output)

    def predict_from_seq(
        self,
        seq: str,
        nce: float,
        as_spectrum=False,
    ) -> PredictionResults | AnnotatedPeptideSpectrum:
        """
        Predict_from_seq Predicts spectra from a sequence as a string.

        Utility method that gets a sequence as a string, encodes it internally
        to the correct input form and outputs the predicted spectra.

        Note that the spectra is not decoded as an output, please check
        `elfragmentador.encoding_decoding.decode_fragment_tensor` for the
        decoding.

        The irt is scaled by 100 and is in the Biognosys scale.

        TODO: consider if the output should be decoded ...

        Parameters:
            seq (str):
                Sequence to use for prediction, supports modifications in the form
                of S[PHOSPHO], S[+80] and T[181]
            nce (float):
                Normalized collision energy to use during the prediction
            as_spectrum (bool, optional):
                Wether to return a Spectrum object instead of the raw tensor predictions
                (Default value = False)

        Returns:
          PredictionResults: A named tuple with two named results; irt and spectra
          Spectrum: A spectrum object with the predicted spectrum

        Examples:
            >>> import pytorch_lightning as pl
            >>> pl.seed_everything(42)
            42
            >>> my_model = PepTransformerBase(num_fragments=CONFIG.num_fragment_embeddings) # Or load the model from a checkpoint
            >>> _ = my_model.eval()
            >>> my_model.predict_from_seq("MYPEPT[U:21]IDEK/3", 27)
            PredictionResults(irt=tensor(..., grad_fn=<PermuteBackward0>), \
            spectra=tensor([...], grad_fn=<PermuteBackward0>))
            >>> out = my_model.predict_from_seq("MYPEPT[U:21]IDEK/3", 27, \
            as_spectrum=True)
            >>> type(out)
            <class 'ms2ml.spectrum.AnnotatedPeptideSpectrum'>
            >>> # my_model.predict_from_seq("MYPEPT[U:21]IDEK/3", 27)
        """  # noqa

        in_batch = Tensorizer().convert_string(data=seq, nce=nce)
        device = next(self.parameters()).device
        in_batch_dict = {k: v.clone().to(device) for k, v in in_batch._asdict().items()}

        out = self.forward(**in_batch_dict)
        logger.debug(out)

        if as_spectrum:
            spec = DeTensorizer.make_spectrum(
                seq=in_batch.seq,
                mod=in_batch.mods,
                charge=in_batch.charge,
                fragment_vector=out.spectra,
                irt=out.irt,
            )
            out = spec

        return out
