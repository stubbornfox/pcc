import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import tensorflow as tf
import sklearn.cluster as skcluster
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn import tree as sktree
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import ensemble
from  sklearn.metrics import confusion_matrix, plot_confusion_matrix
import math
import pickle
from collections import Counter
N = 10
def euclidean_distance(a, b):
  dist = np.linalg.norm(a - b)
  return dist

def build_features(cluster_centers,cluster_class_prob, image_activations, path_features):
  if not os.path.isdir("preprocess-data/features/"):
    os.makedirs("preprocess-data/features/")
  result = []
  num = len(image_activations)
  i = 0

  # with open('kmean.pkl', 'rb') as file:
  #   kmeans = pickle.load(file)
  # # print(set(kmeans.labels_))
  # save_kmean = []
  # for item in image_activations:
  #   id, image = item
  #   features = []
  #   a = kmeans.predict(image)
  #   # print('Before', a)
  #   # a = np.intersect1d(a, cluster_class_prob)
  #   # print('After', a)
  #   # save_kmean.append(a)
  #   temp = np.zeros((800,), dtype=int)
  #   count = Counter(a)
  #   for b in a:
  #     temp[b] = 1 * count[b]
  #   # save_kmean.append(temp)
  #   # print(temp)
  #   # for center, c_p in zip(cluster_centers, cluster_class_prob):
  #   #   c_class, c_prob = c_p
  #   #   min_distance = float('inf')
  #   #   temp_c = 0
  #   result.append((id, temp))
  # np.savez_compressed(path_features, arr=result)
  for item in image_activations:
    id, image = item
    features = []
    for center, c_p in zip(cluster_centers, cluster_class_prob):
      c_class, c_prob = c_p
      min_distance = float('inf')
      temp_c = 0
      for act in image:
        distance = euclidean_distance(act, center)
        # distance = cosine_simirlar(act, center)
        if min_distance > distance:
           min_distance = distance
      if (min_distance <= 20):
        temp_c = c_class
        print(temp_c)
      features.append(temp_c)
      # features.append(min(map(lambda act: euclidean_distance(act, center), image)))
    result.append((id, features))
    i += 1
    print("{}/{}".format(i, num))
  np.savez_compressed(path_features, arr=result)


def read_features(path_features):
  features = np.load(path_features, allow_pickle=True)['arr']
  ids = []
  X = []
  for item in features:
    id, fea = item
    ids.append(id)
    X.append(fea)
  X = np.array(X)
    # np.around(np.array(X)).astype(int)
  # X = np.array([a(xi) for xi in X])
  # X = threshold(X)
  return ids, X

def read_data(path_activations, keys):
  image_activations = []
  for act_file in glob.glob(path_activations):
    id = act_file.split('/')[-1].split('.')[0]

    if int(id) in keys:
      image_activations.append((id, np.load(act_file)['arr']))
  return image_activations

def read_train_data(keys):
  return read_data("preprocess-data/activations/*", keys)

def read_test_data(keys):
  return read_data("preprocess-data/test/activations/*", keys)

def read_image_class_labels():
  path_images_class_labels = "data/CUB_200_2011/image_class_labels.txt"
  result={}

  with open(path_images_class_labels, 'r') as f:
    for line in f:
      iline = line.strip('\n').split(',')[0]
      id, label = iline.split(' ')
      if int(label) <= N:
        result[id] = label
  return result

def read_cluster(n_cluster = 10):
  cluster_files = "preprocess-data/clusters/{}.npz".format(n_cluster)
  cluster_load = np.load(cluster_files, allow_pickle=True)
  acts = cluster_load['center_images_acts']

  temp = []
  c_counters = cluster_load['counters']
  for cc in c_counters:
    c, prob = cc.most_common(1)[0]
    temp.append((c, prob/sum(cc.values())))
  return acts, temp

def build_tree(n_cluster = 10):
  n_class = 5
  start_time = time.time()
  train_path_features = "preprocess-data/features/kmean_{}.npz".format(n_cluster)
  test_path_features = "preprocess-data/test/features/kmean_{}.npz".format(n_cluster)

  if not os.path.isdir("preprocess-data/test/features"):
    os.makedirs("preprocess-data/test/features")
  labels_dict = read_image_class_labels()
  keys = np.fromiter(labels_dict.keys(), dtype=int)
  image_activations = read_train_data(keys)
  test_image_activations = read_test_data(keys)
  cluster_activations, cluster_class_prob = read_cluster(n_cluster)
  i = 0
  prob = {}
  index = []
  # print('cluster_class_prob', len(cluster_class_prob))
  for c_p in cluster_class_prob:
    c, p = c_p
    if p >= 0.8:
      prob[c] = p
      index.append(i)
    i += 1
  index = np.array(index).astype(int)

  # print('cluster_class_prob > 50', len(index))
  cluster_activations = np.array(cluster_activations)[index]
  cluster_class_prob = np.array(cluster_class_prob)[index]
  print(len(index))
  # print(len(image_activations))
  # print(len(test_image_activations))
  # return
  # cluster_class_prob = index

  if not os.path.exists(train_path_features):
    print('Build Features Train')
    build_features(cluster_activations,cluster_class_prob, image_activations, train_path_features)

  if True or not os.path.exists(test_path_features):
    print('Build Features Test')
    build_features(cluster_activations, cluster_class_prob, test_image_activations, test_path_features)

  ids, X_train = read_features(train_path_features)
  id_test, X_test = read_features(test_path_features)


  y_train = []

  for id in ids:
    y_train.append(labels_dict[id])

  y_test = []
  for id in id_test:
    y_test.append(labels_dict[id])

  # print(ids[:10])
  # print(y_train[:10])
  # print(id_test[:10])
  # print(y_test[:10])
  # X_train = np.around(np.array(X_train))
  print(X_train.shape)
  from sklearn.feature_selection import VarianceThreshold
  # sel = VarianceThreshold(threshold=(.95 * (1 - 0.95)))
  # X_train = sel.fit_transform(X_train)
  # print(X_train.shape)
  # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state=42)
  print('Build Tree')
  from sklearn.feature_selection import SelectFromModel
  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train).astype(int)
  y_test = np.array(y_test).astype(int)
  # print(y_test)
  # test_005 = np.where(y_test == 1)[0]
  # train_005 = np.where(y_train == 1)[0]
  # np.savetxt('train_features_exa.csv', X_train[test_005])
  # np.savetxt('test_features_exa.csv', X_test[test_005])
  # print(test_005)
  # print(train_005)
  # print(X_train[train_005])
  # for xtest in X_test[test_005][:5]:
  #   print(xtest)
  # print(X_test[test_005])
  count = 0
  # for x,y in zip(X_train[:4000],y_train[:4000]):
  #   a = np.where(x != -1)[0]
  #   if len(a) > 0:
  #     print(x[a], y)
  #     # print(x[a])
  #   #   print(a, y)
  #     count += 1
  # print(count)
  # print(X_train[:5])
  # print(X_test[:5])
  # print(y_train[:5])
  # print(y_test[:5])
  for maxd in [1]:
    # clf = sktree.DecisionTreeClassifier(max_depth=120)
    from sklearn.feature_selection import SelectFromModel
    clf = ensemble.ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf = clf.fit(X_train, y_train)
    # print('depth', clf.get_depth())
    # print('leaves', clf.get_n_leaves())

    print("Accuracy Score on Train:")
    print(clf.score(X_train, y_train, sample_weight=None))
    print("Accuracy Score on Test:")
    print(clf.score(X_test, y_test, sample_weight=None))
    x_path = "result_{}_birds.csv".format(N)
    np.savetxt(x_path, np.array(confusion_matrix(y_test, clf.predict(X_test))).astype(int), fmt='%5.0f')
    print("--- %s seconds ---" % (time.time() - start_time))

build_tree(n_cluster=9000)
