from utils.configuration import Configuration
from utils.dataset import Dataset

from os.path import realpath, dirname, join

configuration = Configuration(
    num_classes=200,
    num_clusters=1000,
    cluster_accuracy_threshold=25,
)

dataset = Dataset(
    base_path=join(realpath(dirname(__file__)), 'data', 'CUB_200_2011'),
    configuration=configuration,
)
