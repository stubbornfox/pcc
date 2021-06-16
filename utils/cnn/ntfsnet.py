from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from utils.cnn.utils import device

class NtsNetWrapper:
    """
    An abstraction layer to use the ntsnet more easily.

    We only care about the dropout, activation layer and the final
    classification result of the network, which is a bit tedious to get from
    ntsnet. We build this wrapper, so that the overall code in
    `interpret_segments.py` is easier to understand.
    """

    # Instance of https://pytorch.org/hub/nicolalandro_ntsnet-cub200_ntsnet
    model: Module
    _last_dropout: Tensor = [None]
    _last_activation: Tensor = [None]

    def __init__(self):
        self.model = torch.hub.load(
            'nicolalandro/ntsnet-cub200',
            model='ntsnet',
            pretrained=True,
            **{
                'topN': 6,
                'device': device(),
                'num_classes': 200
            }
        )
        self.model.eval()

    def interpret(self, input: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dropout_layer = self.model.pretrained_model.dropout
        dropout_registration = dropout_layer.register_forward_hook(
            lambda module, input, dropout: self._safe_dropout(dropout)
        )

        activation_layer = self.model.pretrained_model
        activation_registration = activation_layer.register_forward_hook(
            lambda module, input, activation: self._safe_activation(activation)
        )

        result = self.model(input)
        predicted_bird_id = self._extract_predicted_bird_id(result)

        dropout = self._last_dropout.detach().numpy()
        dropout_registration.remove()

        pre_trained_model_output = self._last_activation.detach().numpy()
        activation_registration.remove()

        return pre_trained_model_output, dropout, predicted_bird_id

    def _extract_predicted_bird_id(self, result):
        # https://pytorch.org/hub/nicolalandro_ntsnet-cub200_ntsnet/#example-usage
        concat_logits = result[3]

        _, predict = torch.max(concat_logits, 1)
        return predict.item() + 1

    def _safe_dropout(self, dropout):
        # The pretrained model gets called _twice_ in ntsnet, see
        # https://github.com/zhouyuangan/NTS-Net/blob/master/core/model.py#L51,L71
        #
        # We only care about the dropout of the _first_ call, as the second
        # one is not passed the raw input image
        if len(dropout) == 1:
            self._last_dropout = dropout[0]

    def _safe_activation(self, activation):
        # TODO: Why only the first?
        #       Its shape is (1, 200), so this already looks like a
        #       classification result?
        relevant_activation = activation[0]

        # The pretrained model gets called _twice_ in ntsnet, see
        # https://github.com/zhouyuangan/NTS-Net/blob/master/core/model.py#L51,L71
        #
        # We only care about the activations of the _first_ call, as the second
        # one is not passed the raw input image
        if len(relevant_activation) == 1:
            self._last_activation = relevant_activation
