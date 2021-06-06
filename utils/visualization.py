from __future__ import annotations

from os.path import join

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from config import dataset, configuration

from steps.s1_build_segments import load_segments_of
from steps.s4_discover_concepts import load_concepts, global_index_mapping


def view_source_image(image_id: int, resize=False) -> None:
    image_names_per_id = dict(dataset.image_id_path_pairs())
    image_name = image_names_per_id[image_id]
    path = dataset.path(join('images', image_name))
    image = Image.open(path)

    if resize:
        shape = configuration.image_shape
        image.resize(shape, Image.BILINEAR)

    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()


def view_segments_of(image_id: int) -> None:
    segments = load_segments_of(image_id)

    fig, ax = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)
    index = 0

    for segment in segments:
        x, y = int(index / 4), (index % 4)

        index += 1
        image = Image.fromarray(segment.astype(np.uint8))
        ax[x, y].imshow(image)

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


def view_concepts():
    concepts = load_concepts(configuration)
    image_ids, _ = dataset.train_test_image_ids()
    index_mapping = global_index_mapping(image_ids)

    for concept_id, k_nearest_concept_indices, center, cluster_id in concepts:
        segments = []
        index_mappings = index_mapping[k_nearest_concept_indices[:10]]
        for _, image_id, local_segment_id in index_mappings:
            segments_of_image = load_segments_of(image_id)
            segment = segments_of_image[local_segment_id]
            segments.append(segment)

        index = 0

        fig, ax = plt.subplots(6, 7, figsize=(10, 10), sharex=True, sharey=True)
        for segment in segments:
            x, y = int(index / 7), (index % 7)
            image = Image.fromarray(segment)
            ax[x, y].imshow(image)
            index += 1

        for segments in ax.ravel():
            segments.set_axis_off()

        plt.tight_layout()
        plt.show()
