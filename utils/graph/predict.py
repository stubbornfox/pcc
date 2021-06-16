from __future__ import annotations

from collections import defaultdict

from PIL.Image import fromarray, Image

from steps.s1_build_segments import load_segments_of
from steps.s2_interpret_segments import load_activations_of
from steps.s3_cluster_segments import load_cluster_model
from steps.s4_discover_concepts import load_concepts, global_index_mapping
from utils.visualization import view_source_image,view_segments_of, view_concepts_of
from steps.s5_build_decision_tree import _measure_cluster_similarity
from utils.dataset import Dataset
import numpy as np
from utils.similarity import cosine_similarity
from collections import Counter
from steps.s4_discover_concepts import load_concepts, global_index_mapping
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


from collections import defaultdict

# initializing dict with lists

def guess_bird(image_id, dataset: Dataset, index_mapping, classes_per_image_id):
    concepts = load_concepts(dataset.configuration)
    acts = np.array(load_activations_of(image_id), dtype='float')
    kmean_model = load_cluster_model(dataset.configuration)
    cluster_labels = kmean_model.predict(acts)

    has_concept_clusters = []
    cluster_ids = []
    # for cluster_label in cluster_labels:
    #     print(cluster_label)
    #     view_concepts_of(cluster_label)
    # for _, k_nearest_concept_indices, _, cluster_id in concepts:
    #     cluster_ids.append(cluster_id)
    #     if cluster_id in cluster_labels:
    #         has_concept_clusters.append(cluster_id)
    # return len(has_concept_clusters) > 0
    similars = _measure_cluster_similarity(acts, concepts)
    relevants_concept = (-similars).argsort()[:10]
    segments = load_segments_of(image_id)
    results = []
    new_dict = defaultdict(list)
    for i_concept in relevants_concept:
        index = 0
        for act in acts:
            if cosine_similarity(act, concepts[i_concept][2]) == similars[i_concept]:
                results.append(index)
                new_dict[index].append(i_concept)
                # view_concepts_of(concepts[i_concept][-1])
                # view_segments_of(image_id, [index])
                # print(concepts[i_concept][2])
            index += 1
                # load_segments_of
            # view_concepts_of(concepts[i_concept][-1])
    # view_segments_of(image_id)
    # print(results)
    b = Counter(results)
    most_common = b.most_common(3)
    most_index = [most_common[0][0]]
    # most_index.append(most_common[1][0])
    # most_index.append(most_common[2][0])
    # most_index.append(most_common[3][0])
    view_segments_of(image_id, most_index)
    most_index = np.array(most_index)

    relevants_concept = np.array(relevants_concept)
    most_act = acts[most_index]

    index = 0

    final_ids = []
    for index in most_index:
        ma = acts[index]
        print(new_dict[index])
        final_concepts = np.array(concepts, dtype=object)[new_dict[index]]
        max_cos = 0
        for _, k_nearest_concept_indices, center, cluster_id in final_concepts[:1]:
            abc = []

            # x = cosine_similarity(center, ma)
            # print(x)
            # if x > max_cos:
            #     max_cos = x
            #     abc = k_nearest_concept_indices
            concept_mapping = index_mapping[k_nearest_concept_indices]
            for _, id, local_segment_id in concept_mapping:
                abc.append(classes_per_image_id[id])
            print(Counter(abc))
            #     local_act = load_activations_of(id)[local_segment_id]
            #     x = cosine_similarity(local_act, ma)
            #     if x > max_cos:
            #         max_cos = x
            #         final_id = id
        final_ids.append(abc)

    # print(image_id, classes_per_image_id[image_id])
    for k_nearest_concept_indices in final_ids:
        uts = []
        concept_mapping = index_mapping[k_nearest_concept_indices]
        for _, id, local_segment_id in concept_mapping:
            uts.append(classes_per_image_id[id])
        # print(Counter(uts))
        a = Counter(uts).most_common(1)[0][0]
        return a
        # return(Counter(uts)[0][0])
    index += 1
    return final_ids