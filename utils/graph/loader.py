from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from os.path import join
from random import choice

from PIL.Image import Image, open

from utils.dataset import Dataset


class GraphDatasetLoader:
    """
    Speeds up the loading of images by id a bit. Also provides functions to
    directly load an image as a data-uri, so it can be displayed in the graph.
    """
    dataset: Dataset
    image_ids_per_class: dict[int, list[int]]
    image_paths_per_id: dict[int, str]
    class_names_per_id: dict[int, str]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.image_paths_per_id = dict(dataset.image_id_path_pairs())
        self.image_ids_per_class = dataset.image_ids_per_class(True, True)
        self.class_names_per_id = dataset.class_names_per_id()

    def class_image_as_data_uri(self, class_id: int, image_id = None) -> str:
        if image_id == None:
            image_ids_of_class = self.image_ids_per_class[class_id]
            image_id = choice(image_ids_of_class)
        image = self._load_image(image_id)

        buffer = BytesIO()
        image.save(buffer, format='png')
        base_64 = b64encode(buffer.getvalue()).decode('utf-8')

        return f'data:image/png;base64,{base_64}', image.size

    def image_as_data_uri(self, image) -> str:
        buffer = BytesIO()
        image.save(buffer, format='png')
        base_64 = b64encode(buffer.getvalue()).decode('utf-8')

        return f'data:image/png;base64,{base_64}', image.size

    def _load_image(self, image_id: int) -> Image:
        image_name = self.image_paths_per_id[image_id]
        path = self.dataset.path(join('images', image_name))

        return open(path)


    def bird_name(self, class_id):
        return self.class_names_per_id[class_id]
