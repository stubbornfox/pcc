from os.path import join

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from config import dataset

from steps.s1_build_segments import load_segments_of


def view_source_image(image_id: int, resize=False) -> None:
    image_names_per_id = dict(dataset.image_id_path_pairs())
    image_name = image_names_per_id[image_id]
    path = dataset.path(join('images', image_name))
    image = Image.open(path)

    if resize:
        shape = dataset.configuration.image_shape
        image.resize(shape, Image.BILINEAR)

    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()


def view_segments_of(image_id: int) -> None:
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
