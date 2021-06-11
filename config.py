from utils.configuration import Configuration
from utils.dataset import Dataset

from os.path import realpath, dirname, join

configuration = Configuration(
    num_classes=10,
    num_clusters=30,
    cluster_accuracy_threshold=25,
)
configuration.use_cropped_images = True

dataset = Dataset(
    base_path=join(realpath(dirname(__file__)), 'data', 'CUB_200_2011'),
    configuration=configuration,
)
