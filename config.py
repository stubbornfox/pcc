from dataclasses import asdict

from utils.configuration import Configuration
from utils.dataset import Dataset

from os.path import realpath, dirname, join

configuration = Configuration(
    num_classes=200,
    num_clusters=1000,
    cluster_accuracy_threshold=50,
)
# configuration.use_cropped_images = True
# configuration.network = 'resnet128'
configuration.network = 'resnet256'

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
