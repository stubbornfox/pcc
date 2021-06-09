import os
from dataclasses import dataclass
import numpy as np
from PIL import Image

@dataclass
class Segment:
    # Id of the bird
    class_id: int

    # Id of the image of the bird
    image_id: int

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
