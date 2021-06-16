from __future__ import annotations

from random import sample

from PIL.Image import fromarray, Image, new as new_image

from steps.s1_build_segments import load_segments_of
from steps.s4_discover_concepts import load_concepts, global_index_mapping
from utils.dataset import Dataset

'''
    Is concerned with computing things like which segments from which birds are 
    in which clusters / concepts, etc.
'''

def get_graph_data(dataset: Dataset, target_id: int = None):
    print('loading concepts from disk...')
    concepts = load_concepts(dataset.configuration)
    image_ids, _ = dataset.train_test_image_ids()
    print('loading concept mappings from disk...')
    index_mapping = global_index_mapping(image_ids)
    classes_per_image_id = dataset.classes_per_image_id(True, True)

    edges = []
    cluster_previews = dict()

    print('finding related clusters...')
    related_cluster_ids = _find_related_clusters(target_id, concepts, index_mapping, classes_per_image_id)
    related_class_ids = set()

    print('computing edges & cluster previews...')
    for _, k_nearest_concept_indices, _, cluster_id in concepts:
        if cluster_id not in related_cluster_ids:
            continue

        concept_mapping = index_mapping[k_nearest_concept_indices]
        cluster_previews[cluster_id] = _build_cluster_preview(concept_mapping)

        for _, image_id, _ in concept_mapping:
            class_id = classes_per_image_id[image_id]
            related_class_ids.add(class_id)
            edges.append((cluster_id, class_id))

    return related_class_ids, related_cluster_ids, edges, cluster_previews


def _find_related_clusters(target_id, concepts, index_mapping, classes_per_image_id) -> set[int]:
    if target_id is None:
        return set([cluster_id for _, _, _, cluster_id in concepts])

    related_cluster_ids = set()
    for _, k_nearest_concept_indices, _, cluster_id in concepts:
        concept_mapping = index_mapping[k_nearest_concept_indices]

        for _, image_id, _ in concept_mapping:
            class_id = classes_per_image_id[image_id]
            if class_id == target_id:
                related_cluster_ids.add(cluster_id)

    return related_cluster_ids


def _build_cluster_preview(local_concept_mapping) -> Image:
    representants = []

    for _, image_id, local_segment_id in sample(list(local_concept_mapping), 9):
        segments_of_image = load_segments_of(image_id)
        segment = segments_of_image[local_segment_id]
        representants.append(fromarray(segment))

    return _image_grid(representants, 3, 3)


def _image_grid(images, rows, cols):
    assert len(images) == rows * cols

    w, h = images[0].size
    grid = new_image('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

