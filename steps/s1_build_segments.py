from __future__ import annotations

from glob import glob
from os.path import join, exists

from PIL import Image
from skimage.segmentation import slic
from tqdm import tqdm

from utils.checkpoints import checkpoint_directory
from utils.configuration import Configuration
from utils.dataset import Dataset
import numpy as np

from utils.paths import ensure_directory_exists


def build_segments(configuration: Configuration, dataset: Dataset) -> None:
    ensure_directory_exists(segments_path())
    images = dataset.image_id_path_pairs()

    if len(images) <= len(glob(segments_path('*.npz'))):
        print(f'Found segments for all {len(images)} images on disk, skipping to the next step...')
        return

    print('Segmenting images...')
    progress = tqdm(images)
    for image_id, file_name in progress:
        progress.desc = file_name.split('/')[-1]
        progress.refresh()

        segment_file_path = segments_path(f'{image_id}.npz')
        if not exists(segment_file_path):
            image = Image.open(file_name)
            shape = configuration.image_shape
            im2arr = np.array(image.resize(shape, Image.BILINEAR))
            im2arr = np.float32(im2arr) / 255.0
            superpixels, patches = _return_superpixels(
                im2arr,
                shape,
                configuration.segment_resolutions,
            )
            _save_segment(segment_file_path, superpixels, patches)


def segments_path(path=''):
    return checkpoint_directory(join('segments', path))


def load_segments_of(image_id) -> np.ndarray:
    return (np.load(segments_path(f'{image_id}.npz'))['arr'] * 255).astype(np.uint8)


def _save_segment(path: str, superpixels, patches) -> None:
    np.savez_compressed(path, arr=superpixels, patches=patches)


def _extract_patch(image, mask, image_shape, average_image_value):
    mask_expanded = np.expand_dims(mask, -1)
    patch = (mask_expanded * image + (
      1 - mask_expanded) * float(average_image_value) / 255)
    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
    image_resized = np.array(image.resize(image_shape, Image.BICUBIC)).astype(float) / 255
    return image_resized, patch


def _return_superpixels(im2arr, image_shape, resolutions):
    unique_masks = []
    for resolution in resolutions:
        param_masks = []
        segments_slic = slic(
            im2arr,
            n_segments=resolution,
            compactness=20,
            sigma=1,
            start_label=1
        )
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
            superpixel, patch = _extract_patch(
                im2arr,
                unique_masks.pop(),
                image_shape,
                117
            )
            superpixels.append(superpixel)
            patches.append(patch)

        return superpixels, patches

def segment_an_image(image, configuration):
  shape = configuration.image_shape
  im2arr = np.array(image.resize(shape, Image.BILINEAR))
  im2arr = np.float32(im2arr) / 255.0
  superpixels, _ = _return_superpixels(
    im2arr,
    shape,
    configuration.segment_resolutions,
  )
  superpixels = np.array(superpixels)
  return (superpixels*255).astype(np.uint8)