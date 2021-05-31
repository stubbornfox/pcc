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
import time

model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                       **{'topN': 6, 'device': 'cpu', 'num_classes': 200})
model.eval()
nstnet_activation = {}

def get_activation(name):
    def hook(model, input, output):
        nstnet_activation[name] = output.detach()
    return hook


def activate():
  start_time = time.time()
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])
  path_segments = "preprocess-data/test/segments/*"
  path_activations = "preprocess-data/test/activations"
  if not os.path.isdir(path_activations):
    os.makedirs(path_activations)

  for file_name in glob.glob(path_segments):
    act = []
    activate_files = file_name.replace('segments', 'activations')
    if os.path.exists(activate_files):
      continue
    print(file_name)
    images = np.load(file_name)['arr']
    for input_image in images:
      x = np.array(input_image * 255).astype(np.uint8)
      input_tensor = preprocess(x)
      input_batch = input_tensor.unsqueeze(0)
      top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(
        input_batch)
      for c in concat_logits:
        act.append(c.detach().numpy())
    np.savez_compressed(activate_files, arr=act)
  print(time.time() - start_time)

def read_image_class_labels():
  path_images_class_labels = "data/CUB_200_2011/image_class_labels.txt"
  result={}
  with open(path_images_class_labels, 'r') as f:
    for line in f:
      iline = line.strip('\n').split(',')[0]
      id, label = iline.split(' ')
      result[id] = label
  return result

def bottle_neck_activate():
  start_time = time.time()
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])
  path_segments = "preprocess-data/segments/*"
  path_activations = "preprocess-data/bottle_neck_activations_1"
  n_class = N_CLASS
  lbl = read_image_class_labels()

  if not os.path.isdir(path_activations):
    os.makedirs(path_activations)

  for file_name in glob.glob(path_segments):
    act = []
    activate_files = file_name.replace('segments', 'bottle_neck_activations_1')
    if not os.path.exists(activate_files):
      continue
    id = file_name.split('/')[-1].split('.')[0]
    if int(lbl[id]) > n_class:
      continue
    images = np.load(file_name)['arr']
    for input_image in images:
      x = np.array(input_image * 255).astype(np.uint8)
      input_tensor = preprocess(x)
      input_batch = input_tensor.unsqueeze(0)

      model.pretrained_model.fc.register_forward_hook(get_activation('fc'))
      output = model.pretrained_model(input_batch)
      cs = nstnet_activation['fc']
      top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(
        input_batch)
      print(cs[0][:10])
      print(concat_logits[0][:10])
      return
      # print(cs)
      # for c in output:
      # print(cs.detach().numpy().shape)
      a = cs.detach().numpy().reshape(-1)
      # print(a.shape)
      act.append(a)
    np.savez_compressed(activate_files, arr=act)
  print(time.time() - start_time)
N_CLASS = 50

def bottle_neck_activate_test():
  start_time = time.time()
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])
  path_segments = "preprocess-data/test/segments/*"
  path_activations = "preprocess-data/test/bottle_neck_activations_1"
  n_class = N_CLASS
  lbl = read_image_class_labels()

  if not os.path.isdir(path_activations):
    os.makedirs(path_activations)

  for file_name in glob.glob(path_segments):
    act = []
    activate_files = file_name.replace('segments', 'bottle_neck_activations_1')
    if os.path.exists(activate_files):
      continue
    id = file_name.split('/')[-1].split('.')[0]
    if int(lbl[id]) > n_class:
      continue
    images = np.load(file_name)['arr']
    for input_image in images:
      x = np.array(input_image * 255).astype(np.uint8)
      input_tensor = preprocess(x)
      input_batch = input_tensor.unsqueeze(0)

      model.pretrained_model.dropout.register_forward_hook(get_activation('dropout'))
      output = model.pretrained_model(input_batch)
      cs = nstnet_activation['dropout'][0]
      # print(cs)
      # for c in output:
      # print(cs.detach().numpy().shape)
      a = cs.detach().numpy().reshape(-1)
      # print(a.shape)
      act.append(a)
    np.savez_compressed(activate_files, arr=act)
  print(time.time() - start_time)

# def a():
#   bottle_neck_activate()
#   bottle_neck_activate_test()
#
# a()

bottle_neck_activate()