from dataclasses import asdict

from utils.configuration import Configuration
from utils.dataset import Dataset

from os.path import realpath, dirname, join

configuration = Configuration(
    num_classes=200,
    num_clusters=2000,
    cluster_accuracy_threshold=0,
    max_depth=None
)
configuration.use_cropped_images = False
# configuration.network = 'resnet128'
# configuration.network = 'resnet256'
# configuration.cluster_activation_type = 'entire_activation'
# configuration.cluster_activation_type = 'meaning_activation'

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
