import sklearn.cluster as skcluster
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import Counter
import pickle

def euclidean_distance(a, b):
  dist = np.linalg.norm(a - b)
  return dist

def read_image_class_labels():
  path_images_class_labels = "data/CUB_200_2011/image_class_labels.txt"
  result={}
  with open(path_images_class_labels, 'r') as f:
    for line in f:
      iline = line.strip('\n').split(',')[0]
      id, label = iline.split(' ')
      result[id] = label
  return result

def cluster(n_clusters = 10):
  start_time = time.time()
  path_activations = "preprocess-data/activations/*"
  path_clusters = "preprocess-data/clusters"
  cluster_files = os.path.join(path_clusters, '{}.npz'.format(n_clusters))
  img_labels = read_image_class_labels()
  if not os.path.isdir(path_clusters):
    os.makedirs(path_clusters)
  # if os.path.exists(cluster_files):
  #   return
  output = []
  access = []
  lbls = []
  print(len(glob.glob(path_activations)))
  for file_np in glob.glob(path_activations):
    id = file_np.split('/')[-1].split('.')[0]
    data = np.load(file_np)['arr']
    segment_index = 0
    l = img_labels[id]
    for segment in data:
      lbls.append(l)
      output.append(segment)
      access.append((file_np, segment_index))
      segment_index += 1
  output = np.array(output)
  access = np.array(access)
  lbls = np.array(lbls)

  km = skcluster.KMeans(n_clusters)
  d = km.fit(output)
  centers = km.cluster_centers_
  labels = km.labels_
  center_images = []
  counters = []
  with open("kmean_{}.pkl".format(n_clusters), "wb") as f:
    pickle.dump(km, f)

  for i in range(max(labels) + 1):
    label_idxs = np.where(labels == i)[0]
    cluster_lbl = lbls[label_idxs]
    counters.append(Counter(cluster_lbl))
    min_distance = float('inf')
    temp_index = None
    for lb_index in label_idxs:
      temp = euclidean_distance(output[lb_index], centers[i])
      if temp < min_distance:
        min_distance = temp
        temp_index = lb_index
    center_images.append(temp_index)
  np.savez_compressed(cluster_files,
                      labels=labels,
                      centers=centers,
                      counters=counters,
                      center_images_index=center_images,
                      center_images_acts=output[center_images],
                      center_images_retrieve=access[center_images])
  print(time.time() - start_time)

def cluster_by_segments(n_clusters = 10):
  start_time = time.time()
  path_activations = "preprocess-data/segments/*"
  path_clusters = "preprocess-data/clusters_by_segment"
  cluster_files = os.path.join(path_clusters, '{}.npz'.format(n_clusters))
  img_labels = read_image_class_labels()
  if not os.path.isdir(path_clusters):
    os.makedirs(path_clusters)
  if os.path.exists(cluster_files):
    return
  output = []
  access = []
  lbls = []
  for file_np in glob.glob(path_activations):
    id = file_np.split('/')[-1].split('.')[0]
    data = np.load(file_np)['arr']
    segment_index = 0
    l = img_labels[id]
    print(l)
    for segment in data:
      lbls.append(l)
      output.append(segment)
      access.append((file_np, segment_index))
      segment_index += 1
  output = np.array(output)
  access = np.array(access)
  lbls = np.array(lbls)

  km = skcluster.KMeans(n_clusters)
  d = km.fit(output)
  centers = km.cluster_centers_
  labels = km.labels_
  center_images = []
  counters = []
  for i in range(max(labels) + 1):
    label_idxs = np.where(labels == i)[0]
    cluster_lbl = lbls[label_idxs]
    counters.append(Counter(cluster_lbl))
    min_distance = float('inf')
    temp_index = None
    for lb_index in label_idxs:
      temp = euclidean_distance(output[lb_index], centers[i])
      if temp < min_distance:
        min_distance = temp
        temp_index = lb_index
    center_images.append(temp_index)
  np.savez_compressed(cluster_files,
                      labels=labels,
                      centers=centers,
                      counters=counters,
                      center_images_index=center_images,
                      center_images_acts=output[center_images],
                      center_images_retrieve=access[center_images])
  print(time.time() - start_time)

def load_center_cluster_image(n_clusters=2):
  path_clusters = "preprocess-data/clusters"
  cluster_files = os.path.join(path_clusters, 'bottle_neck_1_{}.npz'.format(n_clusters))
  center_images = np.load(cluster_files)['center_images_retrieve']
  for center_image in center_images:
    file_np, image_index = center_image
    segment_files = file_np.replace('bottle_neck_activations_1', 'segments')
    print(segment_files)
    segments = np.load(segment_files)['arr']
    imr = np.array(segments[int(image_index)])
    image = Image.fromarray((imr*255).astype(np.uint8))
    plt.imshow(image)
    plt.show()

# cluster(9000)
# load_center_cluster_image(15)
# cluster_by_segments(100)
# cluster(15)

def cluster_by_bn(n_clusters = 10):
  N_CLASS = 50
  start_time = time.time()
  path_activations = "preprocess-data/bottle_neck_activations_1/*"
  path_clusters = "preprocess-data/clusters"
  cluster_files = os.path.join(path_clusters, 'bn_{}.npz'.format(n_clusters))
  img_labels = read_image_class_labels()
  if not os.path.isdir(path_clusters):
    os.makedirs(path_clusters)
  # if os.path.exists(cluster_files):
  #   return
  output = []
  access = []
  lbls = []
  print(len(glob.glob(path_activations)))
  for file_np in glob.glob(path_activations):
    id = file_np.split('/')[-1].split('.')[0]
    data = np.load(file_np)['arr']
    segment_index = 0
    l = img_labels[id]
    if int(l) > N_CLASS: continue
    for segment in data:
      lbls.append(l)
      output.append(segment)
      access.append((file_np, segment_index))
      segment_index += 1
  output = np.array(output)
  access = np.array(access)
  lbls = np.array(lbls)

  km = skcluster.KMeans(n_clusters)
  d = km.fit(output)
  centers = km.cluster_centers_
  labels = km.labels_
  center_images = []
  counters = []
  with open("kmean_bn_{}.pkl".format(n_clusters), "wb") as f:
    pickle.dump(km, f)

  for i in range(max(labels) + 1):
    label_idxs = np.where(labels == i)[0]
    cluster_lbl = lbls[label_idxs]
    counters.append(Counter(cluster_lbl))
    min_distance = float('inf')
    temp_index = None
    for lb_index in label_idxs:
      temp = euclidean_distance(output[lb_index], centers[i])
      if temp < min_distance:
        min_distance = temp
        temp_index = lb_index
    center_images.append(temp_index)
  np.savez_compressed(cluster_files,
                      labels=labels,
                      centers=centers,
                      counters=counters,
                      center_images_index=center_images,
                      center_images_acts=output[center_images],
                      center_images_retrieve=access[center_images])
  print(time.time() - start_time)

cluster_by_bn(250)
# cluster(400)