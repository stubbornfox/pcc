from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Configuration:
    """Global configuration parameters for the algorithm"""

    # The amount of birds that should be used. Is capped at 200.
    num_classes: int

    # The amount of clusters that the segments should be divided into.
    num_clusters: int

    # The minimum percentage of correctly predicted segments a cluster should
    # contain in order to be seen as meaningful.
    # TODO: reference clustering.py here
    cluster_accuracy_threshold: int

    # The shape that each segment will be resized through, before it is fed
    # through the CNN. This depends on the network used.
    image_shape: tuple[int, int] = (448, 448)

    # The different resolutions that are used when segmenting the source images
    segment_resolutions = [15, 50, 80]
