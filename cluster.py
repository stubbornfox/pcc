import numpy as np
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from segmentize import Segment

def _cluster_folder(cluster_id: int) -> str:
    return f'data/CUB_200_2011/dataset/clusters/{str(cluster_id).rjust(3, "0")}'

def cluster_segments(segments: list[Segment]):
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
