from __future__ import annotations

from os.path import join

from utils.paths import data_path
from config import configuration


def checkpoint_directory(path='') -> str:
    dir_name = 'checkpoints'

    if configuration.use_cropped_images:
        dir_name += '_cropped'

    # This means, that the segments are not shared across networks, but as the
    # segmentation does not take that long, we do not include any extra logic to
    # handle this case. If time is of concern, just copy over an existing
    # segment folder.
    if configuration.network != 'ntsnet':
        dir_name += '_' + configuration.network

    return join(data_path(dir_name), path)
