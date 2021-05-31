NUMBER_CLASS = 200
NUMBER_CLUSTER = 500
ERROR_IMAGES = [448, 848, 1401, 2123, 2306, 2310, 3617, 3619, 3780, 5029, 5393, 6321, 6551, 8551, 9322]
import os
from collections import defaultdict, OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
data = defaultdict(list)

def read_image_class_labels():
  path_images_class_labels = "data/CUB_200_2011/image_class_labels.txt"
  result = defaultdict(list)
  with open(path_images_class_labels, 'r') as f:
    for line in f:
      file_id, bird = line.strip('\n').split(',')[0].split(' ')
      bird = int(bird)
      if bird > NUMBER_CLASS: continue
      file_id = int(file_id)
      if file_id not in ERROR_IMAGES:
        result[bird].append(file_id)

  return result

def build_labels_data(images):
  birds = read_image_class_labels()
  convert_lbls = {}
  for bird in birds:
    imgs = birds[bird]
    for img in imgs:
      convert_lbls[img] = bird
  output = []
  for image in images:
    output.append(convert_lbls[image])
  return output

def read_image_path():
  path = 'data/CUB_200_2011'
  path_images = os.path.join(path, 'images.txt')
  path_folder_images = os.path.join(path, 'images')
  images = []

  with open(path_images, 'r') as f:
    for line in f:
      iline = line.strip('\n').split(',')[0]
      id, file_name = iline.split(' ')
      bird  = int(file_name[:3])
      id = int(id)
      if (bird <= NUMBER_CLASS) and (id not in ERROR_IMAGES):
        images.append((id, os.path.join(path_folder_images, file_name)))
  return images

def show_images_segments(image_id):
    path_segments = "v2/data/segments"
    path_image_segments = os.path.join(path_segments, '{}.npz'.format(image_id))
    segments = np.load(path_image_segments)['arr']
    for data in segments:
      show_image(data)

def show_image(data):
  image = Image.fromarray((data * 255).astype(np.uint8))
  plt.imshow(image)
  plt.show()

def load_train_test():
  train = []
  test = []
  path = 'data/CUB_200_2011/train_test_split.txt'
  with open(path, 'r') as f_:
    for line in f_:
      a, b = list(line.strip('\n').split(' '))
      a, b = int(a), int(b)
      if b == 1:
        train.append(a)
      else:
        test.append(a)
  return np.array(train), np.array(test)

def load_activations(train = True):
  path_activations = "v2/data/activations/bn_fc"
  output = []
  img_numbers = []
  images = load_images(train)

  for image in images:
    acts_fn = os.path.join(path_activations, "{}.npz".format(image))
    acts = np.load(acts_fn)['dropouts']
    for act in acts:
      output.append(act)
      img_numbers.append(image)

  images = load_images(False)

  for image in images:
    acts_fn = os.path.join(path_activations, "{}.npz".format(image))
    acts = np.load(acts_fn)['dropouts']
    for act in acts:
      output.append(act)
      img_numbers.append(image)

  output = np.array(output)
  img_numbers = np.array(img_numbers)
  return output, img_numbers

def load_img_activation(img_ids, whole = False):
  output = []
  act_set = 'bn_fc'
  if  whole:
    act_set = 'bn_fc_whole'
  for img_id in img_ids:
    path = "v2/data/activations/bn_fc/{}.npz".format(img_id)
    acts = np.load(path)['dropouts']
    output.append(acts)

  return np.array(output)

def load_img_activation_outputlayer(img_ids, whole = False):
  output = []
  for img_id in img_ids:
    path = "v2/data/activations/bn_fc/{}.npz".format(img_id)
    acts = np.load(path)['arr']
    output.append(acts)

  return np.array(output)

def load_images(train = True):
  birds = read_image_class_labels()

  train_images, test_images = load_train_test()
  keep_images = test_images

  if train:
    keep_images = train_images

  images = []
  for bird in birds:
    images.extend(birds[bird])
  images = np.array(images)
  keep_images = np.array(keep_images)
  final_images = np.intersect1d(images, keep_images)
  final_images = np.sort(final_images)
  return final_images

def load_segments(train = True, concept_idxs = []):
  print('Load Segments')
  path_segments = "v2/data/segments"
  images = load_images(True)

  index, count, len_idxs = 0, 0, len(concept_idxs)
  output = []

  for image in images:
    sgm_fn = os.path.join(path_segments, "{}.npz".format(image))
    sgms = np.load(sgm_fn)['arr']
    for sgm in sgms:
      if index in concept_idxs:
        output.append(sgm)
        count += 1
        if len_idxs == count:
          return np.array(output)
      index += 1
  images = load_images(False)

  for image in images:
    sgm_fn = os.path.join(path_segments, "{}.npz".format(image))
    sgms = np.load(sgm_fn)['arr']
    for sgm in sgms:
      if index in concept_idxs:
        output.append(sgm)
        count += 1
        if len_idxs == count:
          return np.array(output)
      index += 1

  output = np.array(output)
  return output

def load_cluster(cluster_path):
  loaded = np.load(cluster_path)
  return loaded['asg'], loaded['cost'], loaded['centers']

def load_concept(concept_path):
  return  np.load(concept_path, allow_pickle=True)['arr']

def euclidean_distance(a, b):
  dist = np.linalg.norm(a - b)
  return dist

def filter_important_acts(acts, concepts, cluster_model):
  pass

def load_kmean_model():
  with open("v2/data/clusters/kmean_{}_{}.pkl".format(NUMBER_CLUSTER, NUMBER_CLASS), 'rb') as file:
    kmeans = pickle.load(file)
  return kmeans