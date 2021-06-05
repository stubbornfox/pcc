from glob import glob
from os.path import join, exists

import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from steps.s1_build_segments import load_segment
from utils import cnn
from utils.checkpoints import checkpoint_directory
from utils.cnn import NtsNetWrapper
from utils.configuration import Configuration
from utils.dataset import Dataset
from utils.paths import ensure_directory_exists


def interpret_segments(configuration: Configuration, dataset: Dataset) -> None:
    ensure_directory_exists(activation_path())
    ensure_directory_exists(prediction_path())

    model = NtsNetWrapper(cnn.model())
    processing_pipeline = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print('Interpreting segments...')
    for bird_id, image_ids in tqdm(dataset.image_ids_per_class(
        # We can safely use _all_ images here, as we only write the interpreted
        # segments to disk. In the following training steps we will exclude the
        # test data.
        include_test_images=True,
        include_train_images=True,
    ).items(), position=0):
        for image_id in tqdm(image_ids, position=1, leave=False):
            activation_file = activation_path(f'{image_id}.npz')
            prediction_file = prediction_path(f'{image_id}.npz')

            if exists(activation_file) and exists(prediction_file):
                continue

            activations = []
            dropouts = []
            predictions = []
            segments = (load_segment(image_id) * 255).astype(np.uint8)

            with torch.no_grad():
                for segment in segments:
                    input_tensor = processing_pipeline(segment)
                    input_batch = input_tensor.unsqueeze(0)

                    [
                        activation,
                        dropout,
                        predicted_bird_id
                    ] = model.interpret(input_batch)

                    predictions.append(predicted_bird_id)
                    activations.append(activation)
                    dropouts.append(dropout)

            # These two are only separated, as we first only computed the
            # activations and later also needed the predictions. As this
            # training step takes a while, we wanted to preserve backwards
            # compatibility with the files we had already generated.
            np.savez_compressed(
                activation_file,
                arr=activations,
                dropouts=dropouts,
            )
            np.savez_compressed(
                prediction_file,
                arr=predictions,
                correct=np.array(predictions) == bird_id
            )


def activation_path(path=''):
    return checkpoint_directory(join('activations', path))


def prediction_path(path=''):
    return checkpoint_directory(join('predictions', path))


def load_train_activations_from_disk(dataset: Dataset):
    train_image_ids, _ = dataset.train_test_image_ids()
    train_image_ids = set(train_image_ids)

    return list(np.array([
        np.load(file)
        for file
        in glob(activation_path('*.npz'))
        if int(file.removesuffix('.npz')) in train_image_ids
    ]).flatten())


def load_activations_of(image_id) -> np.ndarray:
    return np.load(activation_path(f'{image_id}.npz'))['arr']


def load_correct_predictions_of(image_id) -> np.ndarray:
    return np.load(prediction_path(f'{image_id}.npz'))['correct']