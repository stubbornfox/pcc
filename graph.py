from config import dataset
from utils.graph.framework import display_graph
from utils.graph.graph_data import get_graph_data
from utils.graph.nodes import class_node, edge, cluster_node, class_node_id, cluster_node_id

# This could be the result from a decision tree, but to keep things simple, we
# can directly change it here.
target_class_id = 1
print(f'Explaining classification for class {target_class_id}:')

print('Gathering computed data...')
num_classes = dataset.configuration.num_classes

[
    related_class_ids, related_cluster_ids, edge_id_pairs, cluster_previews,
    # Set the second parameter to None, if you want to load _all_ classes
    # & clusters
] = get_graph_data(dataset, target_class_id)


print('Building class nodes...')
class_nodes = [class_node(class_id) for class_id in related_class_ids]

print('Building cluster nodes...')
cluster_nodes = [
    cluster_node(cluster_id, preview_image)
    for cluster_id, preview_image
    in cluster_previews.items()
]

print('Building edges...')
edges = []
for cluster_id, class_id in edge_id_pairs:
    edges.append(edge(
        source=class_node_id(class_id),
        target=cluster_node_id(cluster_id),
    ))

print('Started graph server!')
display_graph(
    root_id=class_node_id(target_class_id),
    elements=class_nodes + cluster_nodes + edges,
)

