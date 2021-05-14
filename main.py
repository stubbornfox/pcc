from cluster import cluster_segments
from segmentize import segment_source_images

# Step 0 is to download the data set and run `crop_images.py` to crop out the
# parts of the images that actually contain the birds.

# Then we split up each source image into smaller segments from that image and
# enrich it using features detected by googlenet.
segments = segment_source_images()

# Then we cluster the segments based on the detected features
cluster_segments(segments)
