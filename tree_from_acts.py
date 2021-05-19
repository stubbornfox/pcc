from sklearn import tree
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import glob
from segmentize import _extract_patch, _return_superpixels
from PIL import Image
import torch
from torchvision import transforms
import time

train_file_names = []
train_labels = []
train_activations = []

def similarity(a, b):
  dist = np.linalg.norm(a - b)
  return dist

def build_features(cc, activations):
  result = []
  for act in activations:
    features = []
    for center in cc:
      features.append(similarity(act, center))
    result.append(features)
  return np.array(result)

start_time = time.time()
print("--- %s start time prepare data---" % start_time)

cluster_centers = []
with open("centers_2000.csv", "r") as csvfile:
  csvreader = csv.reader(csvfile)
  for row in csvreader:
    cluster_centers.append(row)
cluster_centers = np.asarray(cluster_centers, dtype=np.float64)

with open("whole_training_activation.csv", "r") as csvfile:
  csvreader = csv.reader(csvfile)
  for row in csvreader:
    train_labels.append(row[0])
    train_file_names.append(row[1])
    train_activations.append(row[2:])

train_activations = np.asarray(train_activations, dtype=np.float64)
X_train = build_features(cluster_centers, train_activations)

# Test Preprocessing

test_file_names = []
test_labels = []
test_activations = []
with open("whole_training_activation_test.csv", "r") as csvfile:
  csvreader = csv.reader(csvfile)
  for row in csvreader:
    test_labels.append(row[0])
    test_file_names.append(row[1])
    test_activations.append(row[2:])
test_activations = np.array(test_activations, dtype=np.float64)
X_Test = build_features(cluster_centers, test_activations)
print("--- %s seconds prepare data---" % (time.time() - start_time))

# Tree
start_time = time.time()
print("--- %s start time building tree---" % start_time)
clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(X_train, train_labels)

print('depth', clf.get_depth())
print('leaves', clf.get_n_leaves())
print("Accuracy Score on Train:")
print(clf.score(X_train, train_labels, sample_weight=None))

print("Accuracy Score on Test:")
print(clf.score(X_Test, test_labels, sample_weight=None))
print("--- %s seconds ---" % (time.time() - start_time))