from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def _extract_patch(image, mask, average_image_value=117):
    mask_expanded = np.expand_dims(mask, -1)
    patch = (mask_expanded * image + (
      1 - mask_expanded) * float(average_image_value) / 255)
    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
    image_resized = image.resize((448, 448), Image.BICUBIC)
    # np.array(image.resize((299, 299), Image.BICUBIC)).astype(float) / 255
    # plt.imshow(image_resized)
    # plt.show()
    return image_resized, patch

def _return_superpixels(im2arr):
    n_segmentss = [15]
    n_params = len(n_segmentss)
    unique_masks = []
    for i in range(n_params):
        param_masks = []
        segments_slic = slic(im2arr, n_segments=n_segmentss[i], compactness=20, sigma=1, start_label=1)
        print(segments_slic)
        plt.imshow(mark_boundaries(im2arr, segments_slic))
        plt.show()
        # for s in range(segments_slic.max()):
        #     mask = (segments_slic == s).astype(float)
        #     if np.mean(mask) > 0.001:
        #         unique = True
        #         for seen_mask in unique_masks:
        #             jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
        #             if jaccard > 0.5:
        #                 unique = False
        #                 break
        #         if unique:
        #             param_masks.append(mask)
        # unique_masks.extend(param_masks)
        # superpixels, patches = [], []
        # while unique_masks:
        #     superpixel, patch = _extract_patch(im2arr, unique_masks.pop())
        #     superpixels.append(superpixel)
        #     patches.append(patch)
        #
        # return superpixels, patches

# dataset, image_numbers, patches = [], [], []
# train_patches_save_path = os.path.join('data/CUB_200_2011', 'dataset/train_patches_1/')
# path = 'data/CUB_200_2011/dataset/train_crop/*/*'
# index  = 0
# shape = (448, 448)
# for k in glob.glob(path):
#     img = Image.open(k)
#     im2arr = np.array(img.resize(shape, Image.BILINEAR))
#     # Normalize pixel values to between 0 and 1.
#     im2arr = np.float32(im2arr) / 255.0
#     file_name = k.split('/')[-2]
#     print(k)
#     if not os.path.isdir(train_patches_save_path + file_name):
#         os.makedirs(os.path.join(train_patches_save_path, file_name))
#         index = 0
#     image_superpixels, image_patches = _return_superpixels(im2arr)
#     for superpixel, patch in zip(image_superpixels, image_patches):
#         index += 1
#         # print(superpixel)
#         im = superpixel
#             #Image.fromarray((superpixel * 255).astype(np.uint8))
#         im.save(os.path.join(os.path.join(train_patches_save_path, file_name), str(index)+'.jpg'))
# # sdataset, simage_numbers, spatches = \
# #     np.array(dataset), np.array(image_numbers), np.array(patches)
# # print(sdataset.shape)

def print_sesgments_examples():
    shape = (448, 448)
    k = 'data/CUB_200_2011/images/002.Laysan_Albatross/Laysan_Albatross_0003_1033.jpg'
    img = Image.open(k)
    im2arr = np.array(img.resize(shape, Image.BILINEAR))
    im2arr = np.float32(im2arr) / 255.0
    n_segmentss = [15, 50, 80]
    n_params = len(n_segmentss)
    unique_masks = []
    for i in range(n_params):
        param_masks = []
        segments_slic = slic(im2arr, n_segments=n_segmentss[i], compactness=20, sigma=1, start_label=1)
        plt.imshow(mark_boundaries(im2arr, segments_slic))
        plt.show()
        # for s in range(segments_slic.max()):
        #     mask = (segments_slic == s).astype(float)
        #     if np.mean(mask) > 0.001:
        #         unique = True
        #         for seen_mask in unique_masks:
        #             jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
        #             if jaccard > 0.5:
        #                 unique = False
        #                 break
        #         if unique:
        #             param_masks.append(mask)
        # unique_masks.extend(param_masks)
        # superpixels, patches = [], []
        # while unique_masks:
        #     superpixel, patch = _extract_patch(im2arr, unique_masks.pop())
        #     superpixels.append(superpixel)
        #     patches.append(patch)
        #
        # # return superpixels, patches
        # # print(segments_slic)
        # plt.imshow(mark_boundaries(img, segments_slic))
        # plt.show()

print_sesgments_examples()