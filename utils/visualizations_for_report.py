from __future__ import annotations

from utils.paths import data_path
import numpy as np
from skimage.segmentation import mark_boundaries, slic
from PIL.Image import fromarray, Image

from config import configuration
from utils.visualization import open_source_image

'''
    The functions in this file were used to create images for the overview
    figures of the reports
'''


def output_directory(path='') -> str:
    return data_path(path)


def segments_on_source_image(image_id):
    """
    Basically just a modified version of `view_segments_on_source_image` that
    saves the files for each resolution in a higher resolution.

    Image 614 was used in the report.
    """
    source_image = open_source_image(image_id)
    im2arr = np.float32(np.array(source_image)) / 255.0

    for resolution in configuration.segment_resolutions:
        segments = _create_segments(im2arr, resolution)
        marked = mark_boundaries(im2arr, segments)

        filename = f'segments_of_image_{image_id}_at_resolution_{resolution}.jpg'
        image = fromarray((marked * 255).astype(np.uint8))
        image.save(output_directory(filename))
        print(f'Saved "{output_directory(filename)}"!')


def only_segments(image_id, resolution):
    """Saves each segment as a file with a transparent background"""
    img = open_source_image(image_id)
    im2arr = np.float32(np.array(img)) / 255.0
    segments = _create_segments(im2arr, resolution)

    unique_masks = []
    param_masks = []
    for s in range(segments.max()):
        mask = (segments == s).astype(float)
        if np.mean(mask) <= 0.001:
            continue
        param_masks.append(mask)

    unique_masks.extend(param_masks)
    superpixels = []
    while unique_masks:
        mask = unique_masks.pop()
        mask_expanded = np.expand_dims(mask, -1)
        patch = (mask_expanded * im2arr + (1 - mask_expanded) * float(117) / 255)
        ones = np.where(mask == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        image = fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
        superpixels.append(np.array(image))

    rgba_segments = [
        _rgb_to_rbga(fromarray(rgb_segment), 117)
        for rgb_segment
        in superpixels
    ]

    for i, segment_image in enumerate(rgba_segments):
        filename = f'segment_{i}_of_image_{image_id}_at_resolution_{resolution}.png'
        segment_image.save(output_directory(filename))
        print(f'Saved "{output_directory(filename)}"!')


def _create_segments(image: np.array, resolution: int) -> np.array:
    return slic(image, n_segments=resolution, compactness=20, sigma=1, start_label=1)


def _rgb_to_rbga(rgb_image: Image, transparent_magic_number) -> Image:
    """
    loops over every pixel, and if it is equal to transparent_magic_number,
    we set its opacity to zero.
    """
    copy = np.array(rgb_image.convert('RGBA'))
    for i, row in enumerate(copy):
        for j, cell in enumerate(row):
           if (cell[:2] == transparent_magic_number).all():
               cell[3] = 0

    return fromarray(copy, mode='RGBA')


only_segments(614, 50)
