from __future__ import annotations

import numpy as np

from steps.s2_interpret_segments import load_activations_of, load_correct_predictions_of
from steps.s3_cluster_segments import load_cluster_metrics
from utils.configuration import Configuration
from utils.dataset import Dataset


def discover_concepts(configuration: Configuration, dataset: Dataset) -> list:
    print('Discovering concepts...')
    concepts = []

    # TODO: Why these exact values?
    #       Maybe we can move them to the configuration object?
    min_imgs = 10
    max_imgs = 40

    cluster_ids, all_costs, centers = load_cluster_metrics(configuration)
    train_image_ids, _ = dataset.train_test_image_ids()
    local_concepts = _locate_concept_ids_for(train_image_ids)

    # This will be very fast, so there is no need for a progress bar
    for cluster_id in range(cluster_ids.max() + 1):
        relevant_indices = np.where(cluster_ids == cluster_id)[0]
        num_occurrences = len(relevant_indices)

        if num_occurrences <= min_imgs:
            continue

        costs = all_costs[relevant_indices]
        k_nearest_concept_indices = relevant_indices[np.argsort(costs)[:max_imgs]]

        if _cluster_accuracy_too_low(
            local_concepts,
            k_nearest_concept_indices,
            configuration
        ):
            continue

        concept_id = len(concepts) + 1
        concept = (concept_id, k_nearest_concept_indices, centers[cluster_id], cluster_id)
        concepts.append(concept)

    return concepts

def _cluster_accuracy_too_low(
    segment_image_mappings,
    nearest_concept_indices: np.ndarray,
    configuration: Configuration,
) -> bool:
    num_correct_guesses = 0

    for global_segment_id, image_id, local_segment_id in segment_image_mappings[nearest_concept_indices]:
        correctly_guessed_indices = load_correct_predictions_of(image_id)
        guess_was_correct = correctly_guessed_indices[local_segment_id]
        num_correct_guesses += int(guess_was_correct)

    accuracy = num_correct_guesses / len(nearest_concept_indices)
    threshold = configuration.cluster_accuracy_threshold / 100

    return accuracy < threshold

def _locate_concept_ids_for(image_ids) -> np.ndarray:
    # TODO: Explicitly state what the output of this function is useful for.
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
