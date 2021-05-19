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
import re
import csv
tree = {}
index = 0
with open("tree.csv", "r") as csvfile:
  csvreader = csv.reader(csvfile)
  for row in csvreader:
    index += 1
    tree[index] = []
    ac = row[1][1:-1]
    bac = ac.split("),")
    for bacd in bac:
      d, a= bacd[1:].split(',')
      tree[index].append((d, a))
    # tup = 'a'
    # tree.append(tup)
print(tree)

# for item in tree:
