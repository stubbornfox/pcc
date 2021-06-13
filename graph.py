from PIL.Image import Image

from config import dataset
from utils.graph.framework import display_graph
from utils.graph.graph_data import get_graph_data
from utils.graph.nodes import class_node, edge, cluster_node, class_node_id, cluster_node_id


print('Gathering computed data...')
num_classes = dataset.configuration.num_classes
[cluster_previews, cluster_colors, classes_per_concept] = get_graph_data(dataset)

print('Building class nodes...')
class_nodes = [class_node(class_id + 1) for class_id in range(num_classes)]

print('Building cluster nodes...')
cluster_nodes = [
    cluster_node(cluster_id, preview_image)
    for cluster_id, preview_image
    in cluster_previews.items()
]

print('Building edges...')
edges = []
for cluster_id, class_ids in classes_per_concept.items():
    for class_id in class_ids:
        edges.append(edge(
            source=class_node_id(class_id),
            target=cluster_node_id(cluster_id),
            # This converts an (r,g,b) tuple to the hex-code.
            color='#%02x%02x%02x' % cluster_colors[cluster_id]
        ))

print('Starting graph server...')
display_graph(class_nodes + cluster_nodes + edges)
