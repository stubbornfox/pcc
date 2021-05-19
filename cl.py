from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tcav import utils
import tensorflow as tf
import sklearn.cluster as cluster
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

output = np.random.rand(400*200, 200)
import csv
# y_train = np.array([])
# file_name = np.array([])
# output =np.array([])

file_name = []
y_train = []
output =[]
def calculate_entropy(df_label):
  classes, class_counts = np.unique(df_label, return_counts=True)
  entropy_value = np.sum([(-class_counts[i] / np.sum(class_counts)) * np.log2(class_counts[i] / np.sum(class_counts))
                          for i in range(len(classes))])
  return entropy_value

import time
start_time = time.time()
acc = 0
with open("activation.csv", "r") as csvfile:
  csvreader = csv.reader(csvfile)
  for row in csvreader:
    if not (row):
      continue
    if row[0] != acc:
      acc = row[0]
      print(acc)
    file_name.append(row[1])
    y_train.append(row[0])
    output.append(row[2:])
output = np.array(output)
file_name = np.array(file_name)
y_train = np.array(y_train)

n_clusters = 2000
km = cluster.KMeans(n_clusters)
d = km.fit(output)
centers = km.cluster_centers_
labels = km.labels_
from collections import Counter
result = {}
for i in range(max(labels) + 1):
  label_idxs = np.where(labels == i)[0]
  y_train_for_cluster_i = y_train[label_idxs]
  entropy = calculate_entropy(y_train_for_cluster_i)
  result[i] = (entropy, Counter(y_train_for_cluster_i))
sort_orders = sorted(result.items(), key=lambda x: x[1][0])
print(sort_orders)
print("--- %s seconds ---" % (time.time() - start_time))

def similarity(a, b):
  dist = np.linalg.norm(a - b)
  return  dist/200

def build_tree(sort_orders, xclass, results):
  for item in sort_orders:
    key = item[0]
    counter = item[1][1]
    if xclass in counter.keys():
      prob = counter[xclass] / sum(counter.values())
      if prob >= 0.1:
        results[key] = prob
  return sorted(results.items(), key=lambda x: x[1], reverse=True)

arrays = {}

with open('tree_2000.csv', 'w', newline='') as csvfile:
  spamwriter = csv.writer(csvfile, delimiter=',')
  for i in range(200):
    a = str(i+1).rjust(3,'0')
    ta = (build_tree(sort_orders, a, {}))
    spamwriter.writerow([a,ta])

with open('centers_2000.csv', 'w', newline='') as csvfile:
  spamwriter = csv.writer(csvfile, delimiter=',')
  for c in centers:
    spamwriter.writerow(c)

with open('labels_2000.csv', 'w', newline='') as csvfile:
  spamwriter = csv.writer(csvfile, delimiter=',')
  for c in labels:
    spamwriter.writerow([c])

