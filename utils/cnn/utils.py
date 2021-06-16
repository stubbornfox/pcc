from torch import cuda


def device() -> str:
    return 'cuda' if cuda.is_available() else 'cpu'
