from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from util import *
from activation import *
def print_sesgments_examples(path):
  shape = (448, 448)
  img = Image.open(path)
  im2arr = np.array(img.resize(shape, Image.BILINEAR))
  im2arr = np.float32(im2arr) / 255.0
  n_segmentss = [15, 50, 80]
  n_params = len(n_segmentss)
  fig, ax = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
  img_slics = []
  for i in range(n_params):
    img_slics.append(slic(im2arr, n_segments=n_segmentss[i], compactness=20, sigma=1, start_label=1))

  ax[0].imshow(mark_boundaries(im2arr, img_slics[0]))
  ax[0].set_title("n-segments: 15")
  ax[1].imshow(mark_boundaries(im2arr, img_slics[1]))
  ax[1].set_title('n-segments: 50')
  ax[2].imshow(mark_boundaries(im2arr, img_slics[2]))
  ax[2].set_title('n-segments: 80')

  for a in ax.ravel():
    a.set_axis_off()

  plt.tight_layout()
  plt.show()

# path = 'data/CUB_200_2011/images/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0008_8756.jpg'
# print_sesgments_examples(path)


def print_unique_sesgments(image_id):
  path_segments = "v2/data/segments"

  segment_fn = os.path.join(path_segments, "{}.npz".format(image_id))
  data = (np.load(segment_fn)['arr'] * 255).astype(np.uint8)
  fig, ax = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)
  x, y = 0, 0
  index = 0
  for segement in data:
    x, y = int(index/4), (index % 4)

    index += 1
    print(x, y)
    image = Image.fromarray((segement).astype(np.uint8))
    ax[x, y].imshow(image)

  for a in ax.ravel():
    a.set_axis_off()

  plt.tight_layout()
  plt.show()

#
# print_unique_sesgments(614)
#print_unique_sesgments(642)

# a = load_img_activation_outputlayer([642])[0]
# t = predict(642)
# for b, k in zip(a,t):
#   print(np.argmax(b[0]), k)
# print_unique_sesgments(642)

def print_out_concepts():
  concept_path = "v2/data/important_concept/{}_{}_{}.npz".format(NUMBER_CLUSTER, NUMBER_CLASS, ACCURACY_SEGMENTS)
  cl = load_concept(concept_path)
  img_ids = load_images(True)
  concept_locs = locate_concepts_ids(img_ids)
  for i in cl:
    _, ids, _, _ = i
    print(len(ids))

    a = load_segments(True, ids[:10], concept_locs)

    index = 0

    fig, ax = plt.subplots(6, 7, figsize=(10, 10), sharex=True, sharey=True)
    for segement in a:
      x, y = int(index / 7), (index % 7)
      image = Image.fromarray((segement * 255).astype(np.uint8))
      ax[x, y].imshow(image)
      index += 1

    for a in ax.ravel():
      a.set_axis_off()

    plt.tight_layout()
    plt.show()

# print_out_concepts()
# predict_all_trained_segments()
# predict(imgids)
# train_images, _ = load_train_test()
# local_concepts = locate_concepts_ids(train_images)
print_out_concepts()