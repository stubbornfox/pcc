from __future__ import annotations

from datetime import datetime

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from steps.s2_interpret_segments import load_activations_of
from steps.s4_discover_concepts import load_concepts
from utils.configuration import Configuration
from utils.dataset import Dataset
from utils.similarity import cosine_similarity
from utils.checkpoints import checkpoint_directory
from os.path import join, exists
from utils.paths import ensure_directory_exists

def build_decision_tree(configuration: Configuration, dataset: Dataset):
    print('Generating training data...')
    start = datetime.now()
    concepts = load_concepts(configuration)
    print('num concepts:' + str(len(concepts) ))
    X_train, Y_train, X_test, Y_test = _generate_train_test_data(
        concepts,
        dataset,
    )
    end = datetime.now()
    print(f'Took {end - start}')

    print('Building decision tree...')
    model = RandomForestClassifier(
        min_samples_split=15,
        min_samples_leaf=10,
        n_estimators = 100,
        max_depth = configuration.max_depth,
    )
    model.fit(X_train, Y_train)

    print('Accuracy Scores:')
    print(f'\tTrain:\t{model.score(X_train, Y_train, sample_weight=None)}')
    print(f'\tTest:\t{model.score(X_test, Y_test, sample_weight=None)}')
    _save_tree_model(configuration, model)
    return model


def explore_tree(X_test, estimator, n_nodes, children_left,children_right, feature,threshold,
                suffix='', print_tree= False, sample_id=0, feature_names=None, draw_tree=False):

    if not feature_names:
        feature_names = feature


    # assert len(feature_names) == X.shape[1], "The feature names do not match the number of features."
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    if draw_tree:
        print("The binary tree structure has %s nodes"
              % n_nodes)
    if print_tree:
        print("Tree structure: \n")
        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                      "node %s."
                      % (node_depth[i] * "\t",
                         i,
                         children_left[i],
                         feature[i],
                         threshold[i],
                         children_right[i],
                         ))
            print("\n")
        print()

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    count = 0
    feature_ids = []
    weights = []
    for node_id in node_index:
        tabulation = " "*node_depth[node_id] #-> makes tabulation of each level of the tree
        # tabulation = ""
        count += 1

        if leave_id[sample_id] == node_id:
            if draw_tree:
                print("%s==> Predicted leaf index \n" % (tabulation))
                return feature_ids, weights
            else:
                return count
        if (X_test[sample_id][feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
            feature_ids.append(feature[node_id])
            weights.append(X_test[sample_id][feature[node_id]])
        if draw_tree:
            print("%sdecision id node %s : (X_test[%s, '%s'] (= %s) %s %s)"
                  % (tabulation,
                     node_id,
                     sample_id,
                     feature_names[feature[node_id]],
                     X_test[sample_id][feature[node_id]],
                     threshold_sign,
                     threshold[node_id]))



def predict_bird(bird_ids, configuration):
    model = load_tree_model(configuration)
    concepts = load_concepts(configuration)
    bird_id = bird_ids[:1][0]
    act_index_closest_to_clusters = _act_index_compare_to_cluster(load_activations_of(bird_id), concepts)
    X_train = _build_feature_vectors([bird_id], concepts)
    X_train = np.array(X_train)
    bird_class = model.predict(X_train)
    true_tree = []

    for i, e in enumerate(model.estimators_):
        bird = int(e.predict(X_train)[0])
        if bird_class[0] == bird:
            true_tree.append(i)
    true_tree = np.array(true_tree)
    n_nodes_ = [t.tree_.node_count for t in model.estimators_]
    children_left_ = [t.tree_.children_left for t in model.estimators_]
    children_right_ = [t.tree_.children_right for t in model.estimators_]
    feature_ = [t.tree_.feature for t in model.estimators_]
    threshold_ = [t.tree_.threshold for t in model.estimators_]
    min_nodes = 1000
    min_i = 0

    feature_names = ["Feature_%d" % i for i in range(X_train.shape[1])]
    for i, e in enumerate(model.estimators_):
        if i in true_tree:
            a = explore_tree(X_train, model.estimators_[i], n_nodes_[i], children_left_[i],
                         children_right_[i], feature_[i], threshold_[i],
                         suffix=i, sample_id=0, feature_names=feature_names)
            if a < min_nodes:
                min_nodes = a
                min_i = i
    i = min_i
    print('Good prediction')
    feature_ids, weights = explore_tree(X_train, model.estimators_[i], n_nodes_[i], children_left_[i],
                 children_right_[i], feature_[i], threshold_[i],
                 suffix=i, sample_id=0, feature_names=feature_names, draw_tree=True)
    segments_index = act_index_closest_to_clusters[feature_ids]
    return bird_class, feature_ids, weights, segments_index

def tree_path(path=''):
    return checkpoint_directory(join('tree', path))

def tree_model_file(c: Configuration):
    return tree_path(f'{c.num_clusters}_{c.num_classes}_{c.max_depth}.pkl')

def load_tree_model(configuration: Configuration) -> RandomForestClassifier:
    with open(tree_model_file(configuration), 'rb') as file:
        return pickle.load(file)

def _save_tree_model(configuration: Configuration, model: RandomForestClassifier) -> None:
    ensure_directory_exists(tree_path())
    with open(tree_model_file(configuration), "wb") as file:
        pickle.dump(model, file)

def feature_path(path=''):
    return checkpoint_directory(join('feature', path))

def feature_file(c: Configuration):
    return feature_path(f'{c.num_clusters}_{c.num_classes}.npz')

def _save_features(X_train, X_test, c: Configuration):
    ensure_directory_exists(feature_path())
    np.savez_compressed(feature_file(c), train=X_train, test=X_test)

def _load_features(c: Configuration):
    features = np.load(feature_file(c))
    train, test = features['train'], features['test']
    return train, test

def _generate_train_test_data(concepts, dataset: Dataset):
    train_image_ids, test_image_ids = dataset.train_test_image_ids()

    Y_train, Y_test = dataset.train_test_class_ids()

    if exists(feature_file(dataset.configuration)):
        X_train, X_test = _load_features(dataset.configuration)
    else:
        X_train = _build_feature_vectors(train_image_ids, concepts)
        X_test = _build_feature_vectors(test_image_ids, concepts)
        _save_features(X_train, X_test, dataset.configuration)
    return X_train, Y_train, X_test, Y_test


def _build_feature_vectors(image_ids, concepts):
    activations_per_image = [load_activations_of(id) for id in image_ids]
    result = [
        _measure_cluster_similarity(activations_per_segment, concepts)
        for activations_per_segment
        in activations_per_image
    ]
    return result


def _measure_cluster_similarity(activations_of_image, concepts):
    cluster_distances_per_concept = []
    similarity = cosine_similarity

    for _, _, center, cluster_id in concepts:
        distances = [
            similarity(activations_of_segment, center)
            for activations_of_segment
            in activations_of_image
        ]
        closest_activation = max(distances)
        cluster_distances_per_concept.append(closest_activation)

    return np.array(cluster_distances_per_concept)


def _act_index_compare_to_cluster(activations_of_image, concepts):
    cluster_distances_per_concept = []
    similarity = cosine_similarity

    for _, _, center, cluster_id in concepts:
        distances = [
            similarity(activations_of_segment, center)
            for activations_of_segment
            in activations_of_image
        ]
        closest_activation_index = np.argmax(distances)
        cluster_distances_per_concept.append(closest_activation_index)

    return np.array(cluster_distances_per_concept)