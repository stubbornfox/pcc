from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


class ProtoTree:
    """
    Wrapper around a traditional decision tree, that is able to visualize it's
    decision process.
    """

    tree = RandomForestClassifier()

    def fit(self, X, Y):


        pass

class TreeVisualizer:
    pass