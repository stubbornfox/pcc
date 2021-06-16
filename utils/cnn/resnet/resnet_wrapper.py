from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from utils.cnn.resnet.net_fc import Net
from utils.cnn.utils import device
from utils.configuration import Configuration
from utils.paths import data_path


class ResNetWrapper:
    model: Module

    def __init__(self, configuration: Configuration):
        network = configuration.network

        if not network.startswith('resnet'):
            raise Exception('Resnet is used, but not configured!')

        num_features = int(network.removeprefix('resnet'))

        self.model = _model(configuration.num_classes, num_features)
        pass

    def interpret(self, input: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dropout = np.array(self.model.add_on_layers(self.model.features(input))[0])

        out = self.model(input)
        [predicted_bird_id] = np.array(torch.argmax(out, dim=1))

        return np.array(out), dropout, predicted_bird_id


def _model(num_classes: int, num_features: int):
    args = argparse.Namespace()
    args.net = 'resnet50_inat'
    args.num_features = num_features
    args.disable_pretrained = True

    path_to_saved_model = data_path(f'pre-trained-models/model_state_{num_features}.pth')

    model = Net(3, num_classes, args)
    model = model.to(device=device())
    model.load_state_dict(torch.load(path_to_saved_model, map_location=device()))
    model = model.to(device=device())
    model.eval()

    return model