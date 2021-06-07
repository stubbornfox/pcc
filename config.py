from utils.configuration import Configuration
from utils.dataset import Dataset

from os.path import realpath, dirname, join

configuration = Configuration(
    num_classes=90,
    num_clusters=400,
    cluster_accuracy_threshold=50,
)
dataset = Dataset(
    base_path=join(realpath(dirname(__file__)), 'data', 'CUB_200_2011'),
    configuration=configuration,
)

# | Clusters | Train | Test |
# |==========|=======|======|
# |      200 |    73 |   50 |
# |      400 |    82 |   58 |
# |     1000 |    77 |   52 |
