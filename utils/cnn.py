from typing import Any

import numpy as np
import torch
from torch import cuda, Tensor


def device() -> str:
    return 'cuda' if cuda.is_available() else 'cpu'


_model = None
def model():
    global _model
    if _model is None:
        _model = torch.hub.load(
            'nicolalandro/ntsnet-cub200',
            model='ntsnet',
            pretrained=True,
            **{
                'topN': 6,
                'device': device(),
                'num_classes': 200
            }
        )

    return _model


class NtsNetWrapper:
    """
    An abstraction layer to use the ntsnet more easily.

    We only care about the dropout, activation layer and the final
    classification result of the network, which is a bit tedious to get from
    ntsnet. We build this wrapper, so that the overall code in
    `interpret_segments.py` is easier to understand.
    """

    model: Any
    model: Any
    _last_dropout: torch.Tensor = [None]

    def __init__(self, model):
        self.model = model

    def predict_bird(self, tensor: Tensor) -> int:
        _, _, _, concat_logits, *_ = self.model(tensor)
        _, predict = torch.max(concat_logits, 1)
        return predict.item() + 1

    def interpret(self, tensor: Tensor) -> [np.ndarray, np.ndarray]:
        layer = self.model.pretrained_model.dropout
        registration = layer.register_forward_hook(
            lambda _1, _2, dropout: self._safe_dropout(dropout)
        )
        activation = self.model.pretrained_model(tensor)[0].detach().numpy()
        registration.remove()
        dropout = self._last_dropout.detach().numpy()

        return activation, dropout

    def _safe_dropout(self, dropout):
        self._last_dropout = dropout[0]
