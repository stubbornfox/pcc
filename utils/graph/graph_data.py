from __future__ import annotations

from collections import defaultdict

from PIL.Image import fromarray, Image

from steps.s1_build_segments import load_segments_of
from steps.s4_discover_concepts import load_concepts, global_index_mapping
from utils.dataset import Dataset

'''
    Is concerned with computing things like which segments from which birds are 
    in which clusters / concepts, etc.
'''

def get_graph_data(dataset: Dataset):
    concepts = load_concepts(dataset.configuration)
    image_ids, _ = dataset.train_test_image_ids()
    index_mapping = global_index_mapping(image_ids)
    classes_per_image_id = dataset.classes_per_image_id(True, True)
    classes_per_concept = defaultdict(set)
    images_per_concept = defaultdict(set)
    cluster_previews = dict()

    for _, k_nearest_concept_indices, _, cluster_id in concepts:
        concept_mapping = index_mapping[k_nearest_concept_indices]
        cluster_previews[cluster_id] = _build_cluster_preview(concept_mapping)

        for _, image_id, _ in concept_mapping:
            class_id = classes_per_image_id[image_id]
            images_per_concept[cluster_id].add(image_id)
            classes_per_concept[cluster_id].add(class_id)

    cluster_colors = dict()
    for cluster_id, preview_image in cluster_previews.items():
        width, height = preview_image.size
        dominant_colors = preview_image.getcolors(width * height)
        dominant_colors.sort(key=lambda x: x[0], reverse=True)
        # The first one is always 117, 117, 117 which is our filler grey color,
        # used to fill void left by the segment, as they are not perfectly
        # rectangular
        occurrences, most_popular_color = dominant_colors[1]
        cluster_colors[cluster_id] = most_popular_color

    return cluster_previews, cluster_colors, classes_per_concept


def _build_cluster_preview(local_concept_mapping) -> Image:
    representants = []

    for _, image_id, local_segment_id in local_concept_mapping[:4]:
        segments_of_image = load_segments_of(image_id)
        segment = segments_of_image[local_segment_id]
        representants.append(fromarray(segment))

    # TODO: Here we need to stitch 4 images together to show roughly what the
    #       cluster contains
    return representants[0]
