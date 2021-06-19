from config import dataset
from utils.graph.framework import display_decision_tree
from utils.graph.graph_data import get_cluster_previews, image_segment
from utils.graph.nodes import edge_weight, class_node, edge, cluster_node, class_node_id, cluster_node_id, segment_node_id, segment_node
from steps.s5_build_decision_tree import predict_bird
from steps.s1_build_segments import load_segments_of

target_bird_id = 11750 #11700 #11000 #10000# 9000 #3000 #900 #1 #868, #1000

def draw_decision_path(image_id, dataset):
    classes_per_image_id = dataset.classes_per_image_id(True, True)
    class_id = classes_per_image_id[image_id]
    segments = load_segments_of(image_id)
    predict_class, concept_ids, weights, local_segment_ids = predict_bird(image_id, dataset.configuration)
    print('Predict Class:', predict_class[0])
    print('Actual Class', class_id)

    bird_segments = []

    for local_segment_id in local_segment_ids:
        bird_segments.append(image_segment(local_segment_id, segments))


    cluster_previews = get_cluster_previews(concept_ids, dataset)

    cluster_nodes = [
        cluster_node(cluster_id, preview_image, weight="Similarity: {:.2f}".format(weight))
        for (cluster_id, preview_image), weight
        in zip(cluster_previews.items(), weights)
    ]

    segment_nodes = [
        segment_node(segment_id, bird_segment)
        for segment_id, bird_segment in zip(local_segment_ids, bird_segments)
    ]

    edges = []

    for segment_id, concept_id, weight in zip(local_segment_ids, concept_ids, weights):
        edges.append(edge_weight(
            source=segment_node_id(segment_id),
            target=cluster_node_id(concept_id),
            weight=weight,
        ))

    for segment_id in local_segment_ids:
        edges.append(edge(
            source=class_node_id(class_id),
            target=segment_node_id(segment_id),
        ))


    bird_node = class_node(class_id, image_id)
    label = bird_node['data']['label']
    label += " ∞ Predict class: {}".format(predict_class[0])
    bird_node['data']['weight'] = label
    display_decision_tree(
        root_id=class_node_id(class_id),
        elements=[bird_node] + cluster_nodes + segment_nodes + edges
    )

draw_decision_path(target_bird_id, dataset)
