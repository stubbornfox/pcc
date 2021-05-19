import os
from dataclasses import dataclass
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

@dataclass(unsafe_hash=True)
class Segment:
    # Id of the bird
    class_id: int

    # Id of the image of the bird
    image_id: int

    # Path to the file where the segment originates from
    source_image_path: str

    # Id of the segment of the image of the bird
    segment_id: int

    # Raw pixel values of the source patch
    raw: np.array

    # Feature vector of the image obtained through googlenet. This can easily be
    # replaced by the bottleneck of an auto encoder to ease clustering.
    features: np.array

    def default_filename(self):
        return f'bird_{self.class_id}_image_{self.image_id}_segment_{self.segment_id}.jpg'

    def persist_to_disk(self, folder: str, file_name=None):
        if file_name is None:
            file_name = self.default_filename()
        Image.fromarray(self.raw).save(os.path.join(folder, file_name))

    def view(self):
        """
        Can be used for debugging. Displays the source image and the segment
        side by side.
        """
        _, images = plt.subplots(1, 2)
        images[0].imshow(np.array(Image.open(self.source_image_path)))
        images[1].imshow(self.raw)
        plt.show()
