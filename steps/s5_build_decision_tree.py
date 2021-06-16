from __future__ import annotations

from datetime import datetime

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from steps.s2_interpret_segments import load_activations_of
from steps.s4_discover_concepts import load_concepts
from utils.configuration import Configuration
from utils.dataset import Dataset
from utils.similarity import cosine_similarity


def build_decision_tree(configuration: Configuration, dataset: Dataset):
    print('Generating training data...')
    start = datetime.now()
    concepts = load_concepts(configuration)
    print('num concepts:' + str(len(concepts) ))
    X_train, Y_train, X_test, Y_test = _generate_train_test_data(
        concepts,
        dataset,
    )
    end = datetime.now()
    print(f'Took {end - start}')

    print('Building decision tree...')
    model = DecisionTreeClassifier(
        min_samples_split=10,
        min_samples_leaf=10,
        max_features='sqrt',
    )
    model.fit(X_train, Y_train)

    print('Accuracy Scores:')
    print(f'\tTrain:\t{model.score(X_train, Y_train, sample_weight=None)}')
    print(f'\tTest:\t{model.score(X_test, Y_test, sample_weight=None)}')

    return model


def _generate_train_test_data(concepts, dataset: Dataset):
    train_image_ids, test_image_ids = dataset.train_test_image_ids()

    Y_train, Y_test = dataset.train_test_class_ids()

    X_train = _build_feature_vectors(train_image_ids, concepts)
    X_test = _build_feature_vectors(test_image_ids, concepts)

    return X_train, Y_train, X_test, Y_test


def _build_feature_vectors(image_ids, concepts):
    activations_per_image = [load_activations_of(id) for id in image_ids]

    return [
        _measure_cluster_similarity(activations_per_segment, concepts)
        for activations_per_segment
        in activations_per_image
    ]


def _measure_cluster_similarity(activations_of_image, concepts):
    cluster_distances_per_concept = []
    similarity = cosine_similarity

    for _, _, center, cluster_id in concepts:
        distances = [
            similarity(activations_of_segment, center)
            for activations_of_segment
            in activations_of_image
        ]
        closest_activation = max(distances)
        cluster_distances_per_concept.append(closest_activation)

    return np.array(cluster_distances_per_concept)
