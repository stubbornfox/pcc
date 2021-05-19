from collections import defaultdict

import numpy as np
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from segmentize import Segment
from pandas import DataFrame

class ClusterOfSegments:
    id: int

    segments: list[Segment]
    num_segments: int

    contained_classes: set[int]
    class_distribution: dict[int, int]
    num_distinct_classes: int

    def __init__(self, cluster_id, segments):
        self.id = cluster_id

        self.segments = segments
        self.num_segments = len(segments)

        self.class_distribution = defaultdict(int)
        for segment in segments:
            self.class_distribution[segment.class_id] += 1
        self.contained_classes = set(self.class_distribution.keys())
        self.num_distinct_classes = len(self.contained_classes)

    def contains_class(self, class_id: int) -> bool:
        return class_id in self.contained_classes


def _cluster_folder(cluster_id: int) -> str:
    return f'data/CUB_200_2011/dataset/clusters/{str(cluster_id).rjust(3, "0")}'

def cluster_segments(segments: list[Segment]) -> dict[int, ClusterOfSegments]:
    print("Clustering...")
    features = [segment.features for segment in segments]
    features = np.concatenate(features, 0)
    k_means = KMeans(n_clusters=250)
    k_means.fit(features)

    for cluster_id in k_means.labels_:
        os.makedirs(_cluster_folder(cluster_id), exist_ok=True)

    print("Writing clusters to disk...")
    for segment, cluster_id in tqdm(zip(segments, k_means.labels_)):
        segment.persist_to_disk(_cluster_folder(cluster_id))

    print("Gathering cluster statistics...")
    clusters = {}
    for cluster_id, df in DataFrame(
        data=zip(k_means.labels_, segments),
        columns=['cluster_id', 'segment']
    ).groupby('cluster_id'):
        clusters[cluster_id] = ClusterOfSegments(cluster_id, list(df['segment']))

    return clusters
