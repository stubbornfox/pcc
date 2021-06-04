from os.path import join

import numpy as np

from utils.paths import data_path


def checkpoint_directory(path='') -> str:
    return join(data_path('checkpoints'), path)
