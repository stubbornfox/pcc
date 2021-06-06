import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from steps.s1_build_segments import load_segments_of


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
