import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tcav import utils
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

def euclidean_distance(a, b):
  dist = np.linalg.norm(a - b)
  return dist

def cluster(n_clusters = 10):
  start_time = time.time()
  path_activations = "preprocess-data/activations/*"
  path_clusters = "preprocess-data/clusters"
  cluster_files = os.path.join(path_clusters, '{}.npz'.format(n_clusters))
  if not os.path.isdir(path_clusters):
    os.makedirs(path_clusters)
  output = []
  access = []
  for file_np in glob.glob(path_activations):
    data = np.load(file_np)['arr']
    segment_index = 0
    for segment in data:
      output.append(segment)
      access.append((file_np, segment_index))
      segment_index += 1
  output = np.array(output)
  access = np.array(access)

  km = skcluster.KMeans(n_clusters)
  d = km.fit(output)
  centers = km.cluster_centers_
  labels = km.labels_
  center_images = []
  for i in range(max(labels) + 1):
    label_idxs = np.where(labels == i)[0]
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
                      center_images_index=center_images,
                      center_images_acts=output[center_images],
                      center_images_retrieve=access[center_images])
  print(time.time() - start_time)

def load_center_cluster_image(n_clusters=2):
  path_clusters = "preprocess-data/clusters"
  cluster_files = os.path.join(path_clusters, '{}.npz'.format(n_clusters))
  center_images = np.load(cluster_files, allow_pickle=True)['center_images_retrieve']
  for center_image in center_images:
    file_np, image_index = center_image
    segment_files = file_np.replace('activations', 'segments')
    segments = np.load(segment_files)['arr']
    imr = np.array(segments[int(image_index)])
    print(imr)
    image = Image.fromarray((imr*255).astype(np.uint8))
    plt.imshow(image)
    plt.show()

cluster(10)
load_center_cluster_image(10)