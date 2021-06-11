from __future__ import annotations

from os.path import join

from utils.paths import data_path
from config import configuration


def checkpoint_directory(path='') -> str:
    dir_name = 'checkpoints'
    if configuration.use_cropped_images:
        dir_name += '_cropped'

    return join(data_path(dir_name), path)
