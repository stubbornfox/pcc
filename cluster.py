import pickle
import shutil
from collections import defaultdict
from scipy.stats import entropy

import numpy as np
import os

from matplotlib import pyplot as plt
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
    entropy: float

    def __init__(self, cluster_id, segments):
        self.id = cluster_id

        self.segments = segments
        self.num_segments = len(segments)

        self.class_distribution = defaultdict(int)
        for segment in segments:
            self.class_distribution[segment.class_id] += 1
        self.contained_classes = set(self.class_distribution.keys())
        self.num_distinct_classes = len(self.contained_classes)
        self.entropy = entropy([
            n / self.num_segments
            for n
            in self.class_distribution.values()
        ])

    def contains_class(self, class_id: int) -> bool:
        return class_id in self.contained_classes

    def view(self):
        """
        Can be used for debugging. Displays the first 9 segments in a 3x3 grid.
        """
        figure, images = plt.subplots(3, 3)
        for i in range(9):
            if i >= self.num_segments:
                break

            row = int(i / 3)
            col = i % 3
            segment = self.segments[i]
            cell = images[row, col]

            cell.set_title(f"Bird {segment.class_id}")
            cell.axis('off')
            cell.imshow(segment.raw)

        figure.suptitle(f"Cluster {self.id}")
        plt.tight_layout()
        plt.show()

def _cluster_folder(cluster_id: int = None) -> str:
    if cluster_id is None:
        return 'data/CUB_200_2011/dataset/clusters/'

    return f'data/CUB_200_2011/dataset/clusters/{str(cluster_id).rjust(3, "0")}'

def cluster_segments(segments: list[Segment]) -> dict[int, ClusterOfSegments]:
    checkpoint_file = 'data/checkpoints/kmeans.obj'
    k_means = None

    if not os.path.isfile(checkpoint_file):
        print("Clustering...")
        features = [segment.features for segment in segments]
        features = np.concatenate(features, 0)
        k_means = KMeans(n_clusters=250)
        k_means.fit(features)

        if os.path.exists(_cluster_folder()):
            print("Deleting existing cluster folder to avoid confusion...")
            shutil.rmtree(_cluster_folder())

        for cluster_id in k_means.labels_:
            os.makedirs(_cluster_folder(cluster_id), exist_ok=True)

        print("Writing clusters to disk...")
        for segment, cluster_id in tqdm(zip(segments, k_means.labels_)):
            segment.persist_to_disk(_cluster_folder(cluster_id))
        with open(checkpoint_file, 'wb') as checkpoint:
            pickle.dump(k_means, checkpoint)
    else:
        print(f'Reading clusters from cache ({checkpoint_file})...')
        with open(checkpoint_file, 'rb') as checkpoint:
            k_means = pickle.load(checkpoint)

    print("Gathering cluster statistics...")
    clusters = {}
    for cluster_id, df in DataFrame(
        data=zip(k_means.labels_, segments),
        columns=['cluster_id', 'segment']
    ).groupby('cluster_id'):
        clusters[cluster_id] = ClusterOfSegments(cluster_id, list(df['segment']))

    return clusters
