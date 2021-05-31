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
from util import NUMBER_CLASS, ERROR_IMAGES, read_image_class_labels, read_image_path

model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                       **{'topN': 6, 'device': 'cpu', 'num_classes': 200})
model.eval()
nstnet_activation = {}

preprocess = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def get_activation(name):
    def hook(model, input, output):
        nstnet_activation[name] = output.detach()
    return hook

def acts_for_whole_images():
  IMAGE_SHAPE = (448, 448)
  path_activations = "v2/data/activations/bn_fc_whole"

  if not os.path.isdir(path_activations):
    os.makedirs(path_activations)

  images = read_image_path()
  total = len(images)
  for id, path in images:
    acts_fn = os.path.join(path_activations, "{}.npz".format(id))
    if os.path.exists(acts_fn):
      continue
    print("{}/{}".format(id, total))
    img = Image.open(path)
    im2arr = np.array(img.resize(IMAGE_SHAPE, Image.BILINEAR))
    # Normalize pixel values to between 0 and 1.
    segement = np.float32(im2arr) / 255.0
    input_tensor = preprocess(segement)
    input_batch = input_tensor.unsqueeze(0)

    model.pretrained_model.dropout.register_forward_hook(get_activation('dropout'))
    output = model.pretrained_model(input_batch)[0]
    output = output.detach().numpy()

    cs = nstnet_activation['dropout'][0]
    cs = cs.detach().numpy()
    np.savez_compressed(acts_fn, arr=output, dropouts=cs)

def acts_for_images():
  start_time = time.time()

  path_segments = "v2/data/segments"
  path_activations = "v2/data/activations/bn_fc"

  if not os.path.isdir(path_activations):
    os.makedirs(path_activations)

  birds = read_image_class_labels()

  for bird in birds:
    images = np.array(birds[bird])

    for image in images:
      segment_fn = os.path.join(path_segments, "{}.npz".format(image))
      acts_fn = os.path.join(path_activations, "{}.npz".format(image))
      print(image)
      if os.path.exists(acts_fn):
        continue
      acts = []
      dropouts = []
      data = (np.load(segment_fn)['arr'] * 255).astype(np.uint8)

      for segement in data:
        input_tensor = preprocess(segement)
        input_batch = input_tensor.unsqueeze(0)

        model.pretrained_model.dropout.register_forward_hook(get_activation('dropout'))
        output = model.pretrained_model(input_batch)[0]
        output = output.detach().numpy()
        acts.append(output)

        cs = nstnet_activation['dropout'][0]
        cs = cs.detach().numpy()
        dropouts.append(cs)
      np.savez_compressed(acts_fn, arr=acts, dropouts=dropouts)
      print('Finish:',image)

def predict(image):
  start_time = time.time()

  path_segments = "v2/data/segments"

  segment_fn = os.path.join(path_segments, "{}.npz".format(image))
      # acts_fn = os.path.join(path_activations, "{}.npz".format(image))
      # print(image)
      # if os.path.exists(acts_fn):
      #   continue
      # acts = []
      # dropouts = []
  data = (np.load(segment_fn)['arr'] * 255).astype(np.uint8)
  output = []
  for segement in data:
    input_tensor = preprocess(segement)
    input_batch = input_tensor.unsqueeze(0)

    # model.pretrained_model.dropout.register_forward_hook(get_activation('dropout'))
    top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(input_batch)
    # output = model(input_batch)[0]
    # output = output.detach().numpy()
    _, predict = torch.max(concat_logits, 1)
    pred_id = predict.item()
    print(pred_id)
    output.append(pred_id)
  return output
    # print('bird class:', model.bird_classes[pred_id])


acts_for_images()
# acts_for_whole_images()