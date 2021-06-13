from dataclasses import asdict

from utils.configuration import Configuration
from utils.dataset import Dataset

from os.path import realpath, dirname, join

configuration = Configuration(
    num_classes=20,
    num_clusters=60,
    cluster_accuracy_threshold=25,
)
configuration.use_cropped_images = True

readable = '\n'.join([
    f'{key}: {value}'
    for key, value
    in asdict(configuration).items()
])
print(f'Configuration:\n{readable}')


dataset = Dataset(
    base_path=join(realpath(dirname(__file__)), 'data', 'CUB_200_2011'),
    configuration=configuration,
)
