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


def node_with_image(node_id, label: str, image: str, size=(100, 100)):
    return {
        'data': {
            'id': node_id,
            'label': label,
        },
        'style': _image_node_style(image, size),
    }


def class_node_id(class_id) -> str:
    return f'class-{class_id}'


def class_node(class_id):
    data_uri, size = graph_dataset_loader.class_image_as_data_uri(class_id)

    return node_with_image(
        node_id=class_node_id(class_id),
        label=graph_dataset_loader.bird_name(class_id),
        image=data_uri,
        size=size,
    )


def cluster_node_id(cluster_id) -> str:
    return f'cluster-{cluster_id}'


def cluster_node(cluster_id, cluster_preview_image):
    return node_with_image(
        node_id=cluster_node_id(cluster_id),
        label=f'Cluster {cluster_id}',
        image=cluster_preview_image,
    )


def edge(source: str, target: str, color):
    return {
        'data': {'source': source, 'target': target},
        'style': {'line-color': color}
    }


def _image_node_style(data_uri, size=(100, 100)):
    width, height = size
    largest_extend = max(width, height)

    if (largest_extend > 100):
        aspect_ratio = width / height


    return {
        'width': f'{width}px',
        'height': f'{height}px',
        'background-opacity': 0,
        'background-image': data_uri,
        'background-clip': 'none',
        'background-image-containment': 'over',
        'background-fit': 'contain',
    }
