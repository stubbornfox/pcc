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

model = torch.hub.load('pytorch/vision:v0.9.0', 'googlenet', pretrained=True)
model.eval()
path = 'data/CUB_200_2011/dataset/train_patches/010.Red_winged_Blackbird/*.jpg'
discovered_concepts_dir = 'data/CUB_200_2011/dataset/concepts'
if not os.path.isdir(discovered_concepts_dir):
  os.makedirs(os.path.join(discovered_concepts_dir))
# im2arr = np.array(input_image)
# im2arr = np.float32(im2arr) / 255.0
# input_tensor = torch.from_numpy(im2arr)
# input_batch = input_tensor.unsqueeze(0)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for folder in glob.glob('data/CUB_200_2011/dataset/train_patches/*'):
  output = []
  image_numbers = []
  count = 0
  dataset = []
  data_batch = []
  target_class  = folder.split('/')[-1].split('.')[0]
  discovery_size = len(glob.glob('data/CUB_200_2011/dataset/train_crop/'+target_class+'*/' + '*.jpg'))

  for filename in glob.glob(folder + '/*.jpg'):
    input_image = Image.open(filename)
    dataset.append(np.array(input_image))
    input_tensor = preprocess(input_image)
    data_batch.append(input_tensor)
    # input_batch = input_tensor.unsqueeze(0)
    image_numbers.append(filename)
    count += 1
  dataloader = DataLoader(data_batch, batch_size=100)
  with torch.no_grad():
    for i_batch, sample_batched in enumerate(dataloader):
      output.append(model(sample_batched))

  image_numbers = np.array(image_numbers)
  dataset = np.array(dataset)
  output = np.concatenate(output, 0)
  output = np.reshape(output, [output.shape[0], -1])
  n_clusters = 25
  km = cluster.KMeans(n_clusters)
  d = km.fit(output)
  centers = km.cluster_centers_
  d = np.linalg.norm(
    np.expand_dims(output, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
  asg, cost = np.argmin(d, -1), np.min(d, -1)

  concept_number = 0
  count2 = 0
  for i in range(asg.max() + 1):
    label_idxs = np.where(asg == i)[0]
    if len(label_idxs) > 20:
      print(label_idxs)
      concept_costs = cost[label_idxs]
      concept_idxs = label_idxs[np.argsort(concept_costs)[:40]]
      concept_image_numbers = set(image_numbers[label_idxs])
      highly_common_concept = len(
        concept_image_numbers) > 0.5 * len(label_idxs)
      mildly_common_concept = len(
        concept_image_numbers) > 0.25 * len(label_idxs)
      mildly_populated_concept = len(
        concept_image_numbers) > 0.25 * discovery_size
      cond2 = mildly_populated_concept and mildly_common_concept
      non_common_concept = len(
        concept_image_numbers) > 0.1 * len(label_idxs)
      highly_populated_concept = len(
        concept_image_numbers) > 0.5 * discovery_size
      cond3 = non_common_concept and highly_populated_concept

      if highly_common_concept or cond2 or cond3:
        concept_number += 1
        concept = '{}_concept{}'.format(target_class, concept_number)
        cimage_numbers = image_numbers[concept_idxs]
        images = dataset[concept_idxs]
        image_addresses, patch_addresses = [], []
        patches_dir = os.path.join(discovered_concepts_dir, concept + '_patches')
        images_dir = os.path.join(discovered_concepts_dir, concept)
        if not os.path.isdir(images_dir):
          os.makedirs(os.path.join(images_dir))
        for i in range(len(images)):
          image_name = '0' * int(np.ceil(2 - np.log10(i + 1))) + '{}_{}'.format(
            i + 1, cimage_numbers[i].split('/')[-2])
          patch_addresses.append(os.path.join(patches_dir, image_name + '.jpg'))
          image_addresses.append(os.path.join(images_dir, image_name + '.jpg'))
        for address, image in zip(image_addresses, images):
          Image.fromarray(image).save(address)
          count2 += 1