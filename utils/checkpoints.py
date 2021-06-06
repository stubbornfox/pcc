from __future__ import annotations

from os.path import join

from utils.paths import data_path


def checkpoint_directory(path='') -> str:
    return join(data_path('checkpoints'), path)
