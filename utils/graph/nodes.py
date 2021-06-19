from __future__ import annotations

from config import dataset
from utils.graph.loader import GraphDatasetLoader


'''
    Shorthands for constructing nodes and edges displaying clusters, birds, etc.
    See https://js.cytoscape.org/#style for the available attributes.
    The functions in here are intended to be used to create the `elements` array
    needed by the function exported in ./framework.py
    Also see https://dash.plotly.com/cytoscape for reference.
'''


graph_dataset_loader = GraphDatasetLoader(dataset)


def node_with_image(node_id, label: str, image: str, size=(300, 300), weight=None):
    return {
        'data': {
            'id': node_id,
            'label': label,
            'weight': weight
        },
        'style': _image_node_style(image, size),
    }


def class_node_id(class_id) -> str:
    return f'class-{class_id}'


def class_node(class_id, image_id=None):
    data_uri, size = graph_dataset_loader.class_image_as_data_uri(class_id, image_id)

    return node_with_image(
        node_id=class_node_id(class_id),
        label=graph_dataset_loader.bird_name(class_id) + f' ({class_node_id(class_id)})',
        image=data_uri,
        size=size,
    )

def cluster_node_id(cluster_id) -> str:
    return f'cluster-{cluster_id}'


def cluster_node(cluster_id, cluster_preview_image, weight=None):
    return node_with_image(
        node_id=cluster_node_id(cluster_id),
        label=f'Cluster {cluster_id}',
        image=cluster_preview_image,
        weight=weight
    )

def segment_node_id(segment_id) -> str:
    return f'segment-{segment_id}'

def segment_node(segment_id, bird_segment):
    return node_with_image(
        node_id=segment_node_id(segment_id),
        label=f'Segment {segment_id}',
        image=bird_segment,
    )

def edge(source: str, target: str):
    return {
        'data': {'source': source, 'target': target},
    }

def edge_weight(source: str, target: str, weight = 0):
    if weight > 0.85:
        weight = '✅'
    else:
        weight = '❌'
    return {
        'data': {'source': source, 'target': target, 'weight': weight},
    }


def _image_node_style(data_uri, size):
    width, height = size

    return {
        'width': f'{width}px',
        'height': f'{height}px',
        'background-opacity': 0,
        'background-image': data_uri,
        'background-clip': 'none',
        'background-image-containment': 'over',
        'background-fit': 'contain',
    }
