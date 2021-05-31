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
N = 50
LBL = {}
N_CLUSTER = 100
from collections import Counter

def euclidean_distance(a, b):
  dist = np.linalg.norm(a - b)
  return dist

def build_features(cluster_centers,cluster_class_prob, image_activations, path_features, lbl={}):
  if not os.path.isdir("preprocess-data/features/"):
    os.makedirs("preprocess-data/features/")
  result = []
  temp = []
  num = len(image_activations)
  i = 0



  # for item in image_activations:
  #   id, image = item
  #   features = []
  #   for center, c_p in zip(cluster_centers, cluster_class_prob):
  #     c_class, c_prob = c_p
  #     min_distance = float('inf')
  #     temp_c = 0
  #     for act in image:
  #       distance = euclidean_distance(act, center)
  #       # distance = cosine_simirlar(act, center)
  #       if min_distance > distance:
  #          min_distance = distance
  #
  #     # if (min_distance > 6):
  #     #   min_distance = 100
  #     #   temp_c = c_class
  #     #   print(temp_c)
  #     features.append(min_distance)
  #     # features.append(min(map(lambda act: euclidean_distance(act, center), image)))
  #   result.append((id, features))
  #   i += 1
  #   print("{}/{}".format(i, num))
  # np.savez_compressed(path_features, arr=result)
  accuracy = 0
  cluster_class_prob = np.array(cluster_class_prob)
  for item in image_activations:
    id, image = item
    features = []
    with open("kmean_bn_{}.pkl".format(N_CLUSTER), 'rb') as file:
      kmeans = pickle.load(file)
    a = kmeans.predict(image)
    temp = np.zeros((N + 1,), dtype=float)

    for cc_prob in cluster_class_prob[a]:
      for each_pair in cc_prob:
        c, prob = each_pair
        if int(c) > 50 : continue
        temp[int(c)] += prob
    print(temp)
    if id in lbl:
      print('predict class:', np.argmax(temp), lbl[id])
      if np.argmax(temp) == int(lbl[id]):
        accuracy +=1
    result.append((id, temp))
    i += 1
    print("{}/{}".format(i, num))
  print('accuracy', accuracy)
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
  # X = np.around(X)
  # X = np.array([a(xi) for xi in X])
  # X = threshold(X)
  return ids, X

def read_data(path_activations):
  image_activations = []
  for act_file in glob.glob(path_activations):
    id = act_file.split('/')[-1].split('.')[0]
    image_activations.append((id, np.load(act_file)['arr']))
  return image_activations

def read_train_data():
  return read_data("preprocess-data/bottle_neck_activations_1/*")

def read_test_data():
  return read_data("preprocess-data/test/bottle_neck_activations_1/*")

def read_image_class_labels():
  path_images_class_labels = "data/CUB_200_2011/image_class_labels.txt"
  result={}
  with open(path_images_class_labels, 'r') as f:
    for line in f:
      iline = line.strip('\n').split(',')[0]
      id, label = iline.split(' ')
      result[id] = int(label)
  return result

def read_cluster(n_cluster = 10):
  cluster_files = "preprocess-data/clusters/bn_{}.npz".format(n_cluster)
  cluster_load = np.load(cluster_files, allow_pickle=True)
  acts = cluster_load['center_images_acts']

  temp = []
  c_counters = cluster_load['counters']
  for cc in c_counters:
    a_cc = []
    sumcc = sum(cc.values())
    for c in cc:
      if int(c) > 50: print(c)
      prob = cc[c]
      a_cc.append((c, prob/sumcc))
    temp.append(a_cc)
  return acts, temp

def build_tree(n_cluster = 10):
  start_time = time.time()
  train_path_features = "preprocess-data/features/bt_2050_{}.npz".format(n_cluster)
  test_path_features = "preprocess-data/test/features/bt_2050_{}.npz".format(n_cluster)

  if not os.path.isdir("preprocess-data/test/features"):
    os.makedirs("preprocess-data/test/features")

  image_activations = read_train_data()
  test_image_activations = read_test_data()
  cluster_activations, cluster_class_prob = read_cluster(n_cluster)
  i = 0
  prob = {}
  index = []
  # for c_p in cluster_class_prob:
  #   c, p = c_p
  #   if p >= 0.8:
  #     prob[c] = p
  #     index.append(i)
  #   i += 1
  # index = np.array(index).astype(int)
  # cluster_activations = np.array(cluster_activations)[index]
  # cluster_class_prob = np.array(cluster_class_prob)[index]
  # print(cluster_activations)
  # print(cluster_class_prob)
  labels_dict = read_image_class_labels()
  LBL = labels_dict
  if True or not os.path.exists(train_path_features):
    print('Build Features Train')
    build_features(cluster_activations, cluster_class_prob, image_activations, train_path_features, lbl=labels_dict)

  ids, X_train = read_features(train_path_features)


  y_train = []

  for id in ids:
    y_train.append(labels_dict[id])

  if True or not os.path.exists(test_path_features):
    print('Build Features Test')
    build_features(cluster_activations, cluster_class_prob, test_image_activations, test_path_features, lbl=labels_dict)
  id_test, X_test = read_features(test_path_features)

  y_test = []
  for id in id_test:
    y_test.append(labels_dict[id])
  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train).astype(int)
  y_test = np.array(y_test).astype(int)
  # np.savetxt('train_features.csv', X_train[np.where(y_train == 1)[0]])
  # np.savetxt('test_features.csv', X_test[np.where(y_test == 1)[0]])
  print(X_train.shape)
  from sklearn.feature_selection import VarianceThreshold
  # sel = VarianceThreshold(threshold=(.95 * (1 - 0.95)))
  # X_train = sel.fit_transform(X_train)
  # print(X_train.shape)
  # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state=42)
  print('Build Tree')
  from sklearn.feature_selection import SelectFromModel

  print(y_test[:10])

  clf = sktree.DecisionTreeClassifier()
  clf = ensemble.RandomForestClassifier(random_state=42, n_estimators=100)
  clf = clf.fit(X_train, y_train)
  print("Accuracy Score on Train:")
  print(clf.score(X_train, y_train, sample_weight=None))
  print("Accuracy Score on Test:")
  print(clf.score(X_test, y_test, sample_weight=None))
  x_path = "result_bn_{}_birds.csv".format(N)
  np.savetxt(x_path, np.array(confusion_matrix(y_test, clf.predict(X_test))).astype(int), fmt='%5.0f')
  print("--- %s seconds ---" % (time.time() - start_time))

build_tree(n_cluster=N_CLUSTER)
