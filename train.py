from __future__ import annotations

from steps.s1_build_segments import build_segments
from steps.s2_interpret_segments import interpret_segments
from steps.s3_cluster_segments import cluster_segments
from steps.s4_discover_concepts import discover_concepts
from steps.s5_build_decision_tree import build_decision_tree

from config import configuration, dataset

# Step 1: Segmentation
# ==============================================================================
# In this step we split each source image up into multiple smaller segments.
# These segments are saved to disk and will be re-used in following steps.
build_segments(configuration, dataset)

# Step 2: Interpret the segments
# ==============================================================================
# Now the segments are passed through a convolutional neural network (CNN).
# We keep the output of the last fully connected layer and the final prediction
# result. The former is an interpretation of what the network thinks is
# contained i the image, and the latter can be used in a later step to prune
# unimportant segments (e.g. parts of the background).
interpret_segments(configuration, dataset)

# Step 3: Clustering
# ==============================================================================
# Based on the interpretation of the network gathered from the former step, we
# group similar segments together. These will then represent prototypical
# features, e.g. striped wings or a read beak.
cluster_segments(configuration, dataset)

# Step 4: Concept Discovery
# ==============================================================================
# Now clean our generated clusters, as there will be ones that e.g. only contain
# images of background concepts like greenery, the sky or parts of the ocean.
# This is done by seeing how accurately the network was at identifying the
# target class solely based on each segment of the cluster.
# This accuracy will be higher for segments displaying prototypical parts of a
# bird and lower for random background noise.
concepts = discover_concepts(configuration, dataset)


# Step 5: Decision Tree
# ==============================================================================
# Now we are ready to construct a decision tree based on the clusters.
build_decision_tree(configuration, dataset)
