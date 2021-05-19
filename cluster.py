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

discovered_concepts_dir = 'data/CUB_200_2011/dataset/cluster'

if not os.path.isdir(discovered_concepts_dir):
  os.makedirs(os.path.join(discovered_concepts_dir))
preprocess = transforms.Compose([
    #transforms.Resize((600, 600), Image.BILINEAR),
    # transforms.CenterCrop((448, 448)),
    # transforms.RandomHorizontalFlip(),  # only if train
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True, **{'topN': 6, 'device':'cpu', 'num_classes': 200})
model.eval()
output = []
image_numbers = []
count = 0
dataset = []
data_batch = []
discovery_size = 0
import time
start_time = time.time()
import csv
activated = ['005', '015', '156', '081', '135', '200']
with open('activation.csv', 'a', newline='') as csvfile:
  spamwriter = csv.writer(csvfile, delimiter=',')

  for folder in glob.glob('data/CUB_200_2011/dataset/train_patches_448/*'):
    target_class  = folder.split('/')[-1].split('.')[0]
    print("Process class: {}".format(target_class))
    print("{}/200".format(count))
    count += 1
    if target_class in activated:
      continue
    for filename in glob.glob(folder + '/*.jpg'):
      input_image = Image.open(filename)
      dataset.append(np.array(input_image))
      input_tensor = preprocess(input_image)
      input_batch = input_tensor.unsqueeze(0)
      top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(
        input_batch)
      for c in concat_logits:
        b = [target_class, filename]
        b.extend(c.detach().numpy())
        spamwriter.writerow(b)

        # spamwriter = csv.writer(csvfile, delimiter=',')
        # a = c.numpy()
        # spamwriter.writerow(np.insert(a, 0, target_class))
#
#     dataloader = DataLoader(data_batch, batch_size=7)
#     with torch.no_grad():
#       for i_batch, sample_batched in enumerate(dataloader):
#         top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(
#           sample_batched)
#         output.append(concat_logits)
#         for c in concat_logits:
#           writer.writerow({'class': target_class, filename: image_numbers[i]})
#
#             spamwriter = csv.writer(csvfile, delimiter=',')
#             a = c.numpy()
#             spamwriter.writerow(np.insert(a, 0, target_class))
# output = np.concatenate(output, 0)
# output = np.reshape(output, [output.shape[0], -1])
# print(output.shape)
#
# image_numbers = np.array(image_numbers)
# print("--- %s seconds ---" % (time.time() - start_time))
# n_clusters = 25
# km = cluster.KMeans(n_clusters)
# d = km.fit(output)
# centers = km.cluster_centers_
# d = np.linalg.norm(
#   np.expand_dims(output, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
# asg, cost = np.argmin(d, -1), np.min(d, -1)
# import shutil
# for i in range(asg.max() + 1):
#   label_idxs = np.where(asg == i)[0]
#   if len(label_idxs) > 0:
#     concept = 'cluster_{}'.format(i)
#     images_dir = os.path.join(discovered_concepts_dir, concept)
#     print(images_dir)
#     if not os.path.isdir(images_dir):
#       os.makedirs(os.path.join(images_dir))
#     images = image_numbers[label_idxs]
#     print(concept)
#     for image in images:
#       shutil.copy2(image, images_dir)
#       target_class = image.split('/')[-1].split('.')[0]
#       print(target_class)
#     print('\n')
    # images =  dataset[concept_idxs]
    # for address, image in zip(image_addresses, images):
    #   Image.fromarray(image).save(address)
# centers = km.cluster_centers_
# d = np.linalg.norm(
#   np.expand_dims(output, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
# asg, cost = np.argmin(d, -1), np.min(d, -1)

# concept_number = 0
# count2 = 0
# for i in range(asg.max() + 1):
#   label_idxs = np.where(asg == i)[0]
#   if len(label_idxs) > 20:
#     print(label_idxs)
#     concept_costs = cost[label_idxs]
#     concept_idxs = label_idxs[np.argsort(concept_costs)[:40]]
#     concept_image_numbers = set(image_numbers[label_idxs])
#     highly_common_concept = len(
#       concept_image_numbers) > 0.5 * len(label_idxs)
#     mildly_common_concept = len(
#       concept_image_numbers) > 0.25 * len(label_idxs)
#     mildly_populated_concept = len(
#       concept_image_numbers) > 0.25 * discovery_size
#     cond2 = mildly_populated_concept and mildly_common_concept
#     non_common_concept = len(
#       concept_image_numbers) > 0.1 * len(label_idxs)
#     highly_populated_concept = len(
#       concept_image_numbers) > 0.5 * discovery_size
#     cond3 = non_common_concept and highly_populated_concept
#
#     if highly_common_concept or cond2 or cond3:
#       concept_number += 1
#       concept = '{}_concept{}'.format(target_class, concept_number)
#       cimage_numbers = image_numbers[concept_idxs]
#       images = dataset[concept_idxs]
#       image_addresses, patch_addresses = [], []
#       patches_dir = os.path.join(discovered_concepts_dir, concept + '_patches')
#       images_dir = os.path.join(discovered_concepts_dir, concept)
#       if not os.path.isdir(images_dir):
#         os.makedirs(os.path.join(images_dir))
#       for i in range(len(images)):
#         image_name = '0' * int(np.ceil(2 - np.log10(i + 1))) + '{}_{}'.format(
#           i + 1, cimage_numbers[i].split('/')[-2])
#         patch_addresses.append(os.path.join(patches_dir, image_name + '.jpg'))
#         image_addresses.append(os.path.join(images_dir, image_name + '.jpg'))
#       for address, image in zip(image_addresses, images):
#         Image.fromarray(image).save(address)
#         count2 += 1
print("--- %s seconds ---" % (time.time() - start_time))