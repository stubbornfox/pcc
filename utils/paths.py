from os import makedirs
from os.path import isdir, dirname, realpath, join


def ensure_directory_exists(path: str):
    if not isdir(path):
      makedirs(path)

def data_path(path: str) -> str:
    return join(realpath(dirname(__file__)), '..', 'data', path)
