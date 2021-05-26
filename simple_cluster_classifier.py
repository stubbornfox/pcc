from sklearn.cluster import KMeans

from cluster import ClusterOfSegments


class SimpleClusterClassifier:
    """
    This classifier just looks at which segments are assigned to which clusters
    and based on this makes a classification decision.
    """

    model: KMeans

    def __init__(self, model: KMeans, clusters: ClusterOfSegments):
        self.model = model

    def predict(self, features):
        return self.model.predict(features)