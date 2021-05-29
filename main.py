from pandas import DataFrame

from cluster import cluster_segments
from segmentize import segment_source_images

# Step 0 is to download the data set and run `crop_images.py` to crop out the
# parts of the images that actually contain the birds.

# Then we split up each source image into smaller segments from that image and
# enrich it using features detected by a cnn.
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
    data=[(
        c.id,
        c.num_distinct_classes,
        c.num_segments,
        c.entropy,
    ) for c in clusters.values()],
    columns=['id', 'num_distinct_classes', 'num_segments', 'entropy']
)

df = df \
    .loc[df['num_segments'] > 5] \
    .sort_values(by=['entropy'], ascending=[True])

for id in df.head(10)[['id']].to_numpy():
    clusters[int(id)].view()

chosen_class = 46
appearances = bird_cluster_appearances[chosen_class]
appearances = DataFrame(
    data=[(id, num) for (id, num) in appearances.items()],
    columns=['cluster_id', 'appearances']
)
