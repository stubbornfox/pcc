from collections import defaultdict
from os.path import join

import numpy as np

from utils.configuration import Configuration


class Dataset:
    base_path: str = ''

    configuration: Configuration

    # These seem to not exist or otherwise throw an error on read, so we do not
    # use them.
    error_images = [
        448, 848, 1401, 2123, 2306, 2310, 3617, 3619, 3780, 5029, 5393, 6321,
        6551, 8551, 9322,
    ]

    def __init__(self, configuration: Configuration, base_path: str):
        self.base_path = base_path
        self.configuration = configuration

    def path(self, path):
        return join(self.base_path, path)

    def image_id_path_pairs(self) -> list[tuple[int, str]]:
        """Returns (image_id, filepath) pairs for each image in the dataset"""
        images = []

        with open(self.path('images.txt'), 'r') as f:
            for line in f:
                file_id, file_name = line.strip('\n').split(',')[0].split(' ')
                bird_id = int(file_name[:3])

                contained_in_dataset = bird_id <= self.configuration.num_classes
                is_valid_image = id not in self.error_images
                if contained_in_dataset and is_valid_image:
                    images.append(
                        (int(file_id), self.path(join('images', file_name)))
                    )

        return images

    def _image_bird_id_pairs(
        self,
        include_train_ids: bool,
        include_test_ids: bool,
    ):
        valid_image_ids = set()

        train_image_ids, test_image_ids = self.train_test_image_ids()

        if include_train_ids:
            valid_image_ids.update(train_image_ids)

        if include_test_ids:
            valid_image_ids.update(test_image_ids)

        valid_image_ids.difference_update(set(self.error_images))

        with open(self.path('image_class_labels.txt'), 'r') as file:
            for line in file:
                image_id, bird_id = line.strip('\n').split(',')[0].split(' ')
                bird_id = int(bird_id)
                image_id = int(image_id)

                if image_id in valid_image_ids:
                    yield image_id, bird_id

    def image_ids_per_class(
        self,
        include_train_images: bool,
        include_test_images: bool,
    ) -> dict[int, list[int]]:
        classes = defaultdict(list)

        for image_id, bird_id in self._image_bird_id_pairs(
            include_train_images=include_train_images,
            include_test_ids=include_test_images,
        ):
            classes[bird_id].append(image_id)

        return classes

    def train_test_image_ids(self):
        train = []
        test = []

        with open(self.path('train_test_split.txt'), 'r') as file:
            for line in file:
                image_id, is_training = list(line.strip('\n').split(' '))
                image_id, is_training = int(image_id), int(is_training)
                if is_training == 1:
                    train.append(image_id)
                else:
                    test.append(image_id)

        return np.array(train), np.array(test)

    def load_images(self):
        pass
