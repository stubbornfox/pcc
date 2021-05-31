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

def segment_images(classes = None, shape=IMAGE_SHAPE, crop_path = 'data/CUB_200_2011/dataset/train_crop'):
    path = 'data/CUB_200_2011'
    path_segments = "preprocess-data/multi_res_segments"
    path_images = os.path.join(path, 'images.txt')
    images = []
    if not os.path.isdir(path_segments):
      os.makedirs(path_segments)

    with open(path_images, 'r') as f:
        for line in f:
            iline = line.strip('\n').split(',')[0]
            id, file_name = iline.split(' ')
            if (classes is None) or (file_name[:3] in classes):
                images.append((id, file_name))
    index = 0
    for id, file_name in images:
        full_path_file_name = os.path.join(crop_path, file_name)
        index += 1
        if os.path.exists(full_path_file_name):
            segment_files = os.path.join(path_segments, '{}.npz'.format(id))
            if not os.path.exists(segment_files):
                img = Image.open(full_path_file_name)
                im2arr = np.array(img.resize(shape, Image.BILINEAR))
                im2arr = np.float32(im2arr) / 255.0
                image_superpixels, image_patches = _return_superpixels(im2arr)
                np.savez_compressed(segment_files, arr=image_superpixels)

def show_images_segments(image_id):
    path_segments = "preprocess-data/segments"
    full_path_segments = os.path.join(path_segments, '{}/*'.format(image_id))
    for file_np in glob.glob(full_path_segments):
        data = np.load(file_np)
        image = Image.fromarray((data * 255).astype(np.uint8))
        plt.imshow(image)
        plt.show()

segment_images()

