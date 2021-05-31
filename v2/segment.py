from util import read_image_class_labels, read_image_path

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
IMAGE_SHAPE = (448, 448)

def _extract_patch(image, mask, image_shape=IMAGE_SHAPE, average_image_value=117):
    mask_expanded = np.expand_dims(mask, -1)
    patch = (mask_expanded * image + (
      1 - mask_expanded) * float(average_image_value) / 255)
    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
    image_resized = np.array(image.resize(image_shape, Image.BICUBIC)).astype(float) / 255
    return image_resized, patch

def _return_superpixels(im2arr):
    n_segmentss = [15, 50, 80]
    n_params = len(n_segmentss)
    unique_masks = []
    for i in range(n_params):
        param_masks = []
        segments_slic = slic(im2arr, n_segments=n_segmentss[i], compactness=20, sigma=1, start_label=1)
        for s in range(segments_slic.max()):
            mask = (segments_slic == s).astype(float)
            if np.mean(mask) > 0.001:
                unique = True
                for seen_mask in unique_masks:
                    jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                    if jaccard > 0.5:
                        unique = False
                        break
                if unique:
                    param_masks.append(mask)
        unique_masks.extend(param_masks)
        superpixels, patches = [], []
        while unique_masks:
            superpixel, patch = _extract_patch(im2arr, unique_masks.pop())
            superpixels.append(superpixel)
            patches.append(patch)

        return superpixels, patches

def segment_images(shape=IMAGE_SHAPE):
    path = 'data/CUB_200_2011'
    path_segments = "v2/data/segments"
    images = read_image_path()

    if not os.path.isdir(path_segments):
      os.makedirs(path_segments)

    index = 0
    total = len(images)
    for id, file_name in images:
        print(id, file_name)
        index += 1
        print("{}/{}".format(index, total))
        segment_files = os.path.join(path_segments, '{}.npz'.format(id))
        if not os.path.exists(segment_files):
            img = Image.open(file_name)
            im2arr = np.array(img.resize(shape, Image.BILINEAR))
            im2arr = np.float32(im2arr) / 255.0
            image_superpixels, image_patches = _return_superpixels(im2arr)
            np.savez_compressed(segment_files, arr=image_superpixels, patches=image_patches)

segment_images()
