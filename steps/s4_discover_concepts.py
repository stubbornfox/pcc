from __future__ import annotations

from os.path import join, exists

import numpy as np

from steps.s2_interpret_segments import load_activations_of, load_correct_predictions_of
from steps.s3_cluster_segments import load_cluster_metrics
from utils.checkpoints import checkpoint_directory
from utils.configuration import Configuration
from utils.dataset import Dataset
from utils.paths import ensure_directory_exists


def discover_concepts(configuration: Configuration, dataset: Dataset) -> None:
    ensure_directory_exists(concept_path())
    print('Discovering concepts...')

    if exists(concept_file(configuration)):
        print('Found existing concept file, skipping to the next step...')
        return

    concepts = []

    # TODO: Why these exact values?
    #       Maybe we can move them to the configuration object?
    min_imgs = 10
    max_imgs = 40

    cluster_ids, all_costs, centers = load_cluster_metrics(configuration)
    train_image_ids, _ = dataset.train_test_image_ids()
    index_mapping = global_index_mapping(train_image_ids)

    # This will be very fast, so there is no need for a progress bar
    for cluster_id in range(cluster_ids.max() + 1):
        relevant_indices = np.where(cluster_ids == cluster_id)[0]
        num_occurrences = len(relevant_indices)

        if num_occurrences <= min_imgs:
            continue

        costs = all_costs[relevant_indices]
        k_nearest_concept_indices = relevant_indices[np.argsort(costs)[:max_imgs]]

        if _cluster_accuracy_too_low(
            index_mapping,
            k_nearest_concept_indices,
            configuration
        ):
            continue

        concept_id = len(concepts) + 1
        concept = (concept_id, k_nearest_concept_indices, centers[cluster_id], cluster_id)
        concepts.append(concept)

    _save_concepts(configuration, concepts)

def _cluster_accuracy_too_low(
    index_mapping,
    nearest_concept_indices: np.ndarray,
    configuration: Configuration,
) -> bool:
    num_correct_guesses = 0

    for _, image_id, local_segment_id in index_mapping[nearest_concept_indices]:
        correctly_guessed_indices = load_correct_predictions_of(image_id)
        guess_was_correct = correctly_guessed_indices[local_segment_id]
        num_correct_guesses += int(guess_was_correct)

    accuracy = num_correct_guesses / len(nearest_concept_indices)
    threshold = configuration.cluster_accuracy_threshold / 100

    return accuracy < threshold


def concept_path(path=''):
    return checkpoint_directory(join('concepts', path))


def concept_file(c: Configuration):
    accuracy = c.cluster_accuracy_threshold
    return concept_path(f'{c.num_clusters}_{c.num_classes}_{accuracy}.npz')


def load_concepts(configuration: Configuration):
    return [tuple(row) for row in list(np.load(
        concept_file(configuration),
        allow_pickle=True
    )['concepts'])]


def _save_concepts(configuration: Configuration, concepts):
    np.savez_compressed(
        concept_file(configuration),
        concepts=concepts,
        dtype=object
    )


def global_index_mapping(image_ids) -> np.ndarray:
    """
    Computes indices and offsets, that can be used to map the entries of the
    cluster metrics to their respective images or segments.

    The image ids need to begin at one!
    """

    # This is the "index" of the segment if you would flatten _all_ segments of
    # all images.
    global_segment_id = 0
    output = []

    for image_id in image_ids:
        activations = load_activations_of(image_id)
        for (local_segment_id, _) in enumerate(activations):
            # The local_segment_id is the index of the activation inside its
            # .npz file
            entry = (global_segment_id, image_id, local_segment_id)
            output.append(entry)
            global_segment_id += 1

    return np.array(output)
