import pickle
from os.path import join, exists

import numpy as np
from sklearn.cluster import KMeans

from steps.s2_interpret_segments import load_train_activations_from_disk
from utils.checkpoints import checkpoint_directory
from utils.configuration import Configuration
from utils.dataset import Dataset
from utils.paths import ensure_directory_exists
from utils.similarity import euclidean_distance


def cluster_segments(configuration: Configuration, dataset: Dataset) -> None:
    ensure_directory_exists(cluster_path())
    checkpoint_file = cluster_metrics_file(configuration)

    if exists(checkpoint_file):
        print(f'"{checkpoint_file}" exists, skipping segment clustering...')
        return

    activations = load_train_activations_from_disk(dataset)

    print('Clustering segments...')
    model = KMeans(configuration.num_clusters)
    model.fit(activations)

    print('Saving clusters to disk...')
    _save_cluster_model(configuration, model)
    _save_cluster_metrics(model, activations, checkpoint_file)


def cluster_path(path=''):
    return checkpoint_directory(join('clusters', path))


def cluster_metrics_file(c: Configuration):
    return cluster_path(f'{c.num_classes}_{c.num_classes}.npz')


def cluster_model_file(c: Configuration):
    return cluster_path(f'{c.num_classes}_{c.num_classes}.pkl')


def load_cluster_metrics(configuration: Configuration):
    loaded = np.load(cluster_metrics_file(configuration))
    cluster_ids = loaded['asg']
    cost = loaded['cost']
    centers = loaded['centers']

    return cluster_ids, cost, centers


def load_cluster_model(configuration: Configuration) -> KMeans:
    with open(cluster_model_file(configuration), 'rb') as file:
        return pickle.load(file)


def _save_cluster_model(configuration: Configuration, model: KMeans) -> None:
    with open(cluster_model_file(configuration), "wb") as file:
        pickle.dump(model, file)


def _save_cluster_metrics(
    model: KMeans,
    activations: np.ndarray,
    checkpoint_file: str,
) -> None:
    centers = model.cluster_centers_
    labels = model.labels_
    cost = np.array([
        euclidean_distance(activation, centers[label])
        for activation, label
        in zip(activations, labels)
    ])

    np.savez_compressed(checkpoint_file, asg=labels, cost=cost, centers=centers)
