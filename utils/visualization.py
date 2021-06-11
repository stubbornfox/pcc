from __future__ import annotations

from os.path import join

import numpy as np
from PIL.Image import Image, open
from matplotlib import pyplot as plt
from skimage.segmentation import slic, mark_boundaries

from config import dataset, configuration

from steps.s1_build_segments import load_segments_of
from steps.s4_discover_concepts import load_concepts, global_index_mapping

'''
    The functions in this file can be used to do exploratory analysis on the 
    artifacts created by the steps.
'''


def view_source_image(image_id: int, resize=False) -> None:
    """Displays the source image file"""
    image = open_source_image(image_id)

    if resize:
        shape = configuration.image_shape
        image.resize(shape, Image.BILINEAR)

    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()


def view_segments_of(image_id: int) -> None:
    """Displays all segments of the image, as they were passed to the network"""
    segments = load_segments_of(image_id)

    fig, ax = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)
    index = 0

    for segment in segments:
        x, y = int(index / 4), (index % 4)

        index += 1
        image = Image.fromarray(segment.astype(np.uint8))
        ax[x, y].imshow(image)

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


def view_segments_on_source_image(image_id):
    """This overlays the generated segments on the source image"""
    img = open_source_image(image_id)
    im2arr = np.array(img.resize(configuration.image_shape, Image.BILINEAR))
    im2arr = np.float32(im2arr) / 255.0
    resolutions = configuration.segment_resolutions
    n_params = len(resolutions)
    fig, ax = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
    img_slics = []
    for i in range(n_params):
        img_slics.append(slic(im2arr, n_segments=resolutions[i], compactness=20, sigma=1, start_label=1))

    ax[0].imshow(mark_boundaries(im2arr, img_slics[0], mode='thick'))
    ax[0].set_title("n-segments: 15")
    ax[1].imshow(mark_boundaries(im2arr, img_slics[1], mode='thick'))
    ax[1].set_title('n-segments: 50')
    ax[2].imshow(mark_boundaries(im2arr, img_slics[2], mode='thick'))
    ax[2].set_title('n-segments: 80')

    for a in ax.ravel():
        a.set_axis_off()

    plt.title = f'Image {image_id}'
    plt.tight_layout()
    plt.show()


def view_concepts(num_concepts=10):
    """Displays a figure for each concept found on disk"""
    concepts = load_concepts(configuration)
    image_ids, _ = dataset.train_test_image_ids()
    index_mapping = global_index_mapping(image_ids)

    print(f'Showing up to {num_concepts} concepts...')
    for i, (_, k_nearest_concept_indices, _, _) in enumerate(concepts):
        if i >= num_concepts:
            print(f'There are more concepts, but as num_concepts was reached!')
            break

        segments = []
        index_mappings = index_mapping[k_nearest_concept_indices[:10]]
        for _, image_id, local_segment_id in index_mappings:
            segments_of_image = load_segments_of(image_id)
            segment = segments_of_image[local_segment_id]
            segments.append(segment)

        index = 0

        fig, ax = plt.subplots(6, 7, figsize=(10, 10), sharex=True, sharey=True)
        for segment in segments:
            x, y = int(index / 7), (index % 7)
            image = Image.fromarray(segment)
            ax[x, y].imshow(image)
            index += 1

        for segments in ax.ravel():
            segments.set_axis_off()

        plt.tight_layout()
        plt.show()


def open_source_image(image_id) -> Image:
    image_names_per_id = dict(dataset.image_id_path_pairs())
    image_name = image_names_per_id[image_id]
    path = dataset.path(join('images', image_name))

    return open(path)