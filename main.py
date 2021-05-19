from collections import defaultdict

from pandas import DataFrame

from cluster import cluster_segments
from segmentize import segment_source_images

# Step 0 is to download the data set and run `crop_images.py` to crop out the
# parts of the images that actually contain the birds.

# Then we split up each source image into smaller segments from that image and
# enrich it using features detected by googlenet.
segments = segment_source_images()

# Then we cluster the segments based on the detected features
clusters = cluster_segments(segments)


# Debugging
bird_cluster_appearances = {}
for cluster in clusters.values():
    for class_id, appearances in cluster.class_distribution.items():
        if class_id not in bird_cluster_appearances:
            bird_cluster_appearances[class_id] = {}

        bird_cluster_appearances[class_id][cluster.id] = appearances
df = DataFrame(
    data=[(c.id, c.num_distinct_classes, c.num_segments, list(c.contained_classes)) for c in clusters.values()],
    columns=['id', 'num_distinct_classes', 'num_segments', 'contained_classes']
)
