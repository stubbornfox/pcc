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
def activate():
  start_time = time.time()
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])
  path_segments = "preprocess-data/segments/*"
  path_activations = "preprocess-data/activations"
  if not os.path.isdir(path_activations):
    os.makedirs(path_activations)

  for file_name in glob.glob(path_segments):
    print(file_name)
    act = []
    activate_files = file_name.replace('segments', 'activations')
    if os.path.exists(activate_files):
      continue
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

activate()