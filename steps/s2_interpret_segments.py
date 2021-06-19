from __future__ import annotations

from os.path import join, exists

import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from steps.s1_build_segments import load_segments_of

from utils.checkpoints import checkpoint_directory
from utils.cnn.ntfsnet import NtsNetWrapper
from utils.cnn.resnet.resnet_wrapper import ResNetWrapper
from utils.configuration import Configuration
from utils.dataset import Dataset
from utils.paths import ensure_directory_exists


def interpret_segments(configuration: Configuration, dataset: Dataset) -> None:
    ensure_directory_exists(activation_path())
    ensure_directory_exists(prediction_path())
    use_resnet = configuration.network.startswith('resnet')

    model = ResNetWrapper(configuration) if use_resnet else NtsNetWrapper()
    processing_pipeline = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    bird_progress = tqdm(dataset.image_ids_per_class(
        # We can safely use _all_ images here, as we only write the interpreted
        # segments to disk. In the following training steps we will exclude the
        # test data.
        include_test_images=True,
        include_train_images=True,
    ).items(), position=0, leave=False)

    print('Interpreting segments...')
    for bird_id, image_ids in bird_progress:
        bird_progress.set_description(f'Bird {bird_id}', refresh=True)

        image_process = tqdm(image_ids, position=1, leave=False)
        for image_id in image_process:
            image_process.set_description(f'Image {image_id}', refresh=True)
            activation_file = activation_path(f'{image_id}.npz')
            prediction_file = prediction_path(f'{image_id}.npz')

            if exists(activation_file) and exists(prediction_file):
                continue

            pre_trained_model_outputs = []
            dropouts = []
            predictions = []
            segments = load_segments_of(image_id)

            with torch.no_grad():
                for segment in segments:
                    input_tensor = processing_pipeline(segment)
                    input_batch = input_tensor.unsqueeze(0)

                    [
                        pre_trained_model_output,
                        dropout,
                        predicted_bird_id
                    ] = model.interpret(input_batch)

                    predictions.append(predicted_bird_id)
                    pre_trained_model_outputs.append(pre_trained_model_output)
                    dropouts.append(dropout)

            # These two are only separated, as we first only computed the
            # activations and later also needed the predictions. As this
            # training step takes a while, we wanted to preserve backwards
            # compatibility with the files we had already generated.
            np.savez_compressed(
                activation_file,
                arr=pre_trained_model_outputs,
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

    all_activations = []
    for image_id in train_image_ids:
        activation_file = activation_path(f'{image_id}.npz')
        activation_batch = list(np.load(activation_file)['dropouts'])
        all_activations.extend(activation_batch)
    all_activations = np.array(all_activations)
    all_activations = all_activations[load_train_predictions_from_disk(dataset)]
    return all_activations

def load_train_predictions_from_disk(dataset: Dataset):
    train_image_ids, _ = dataset.train_test_image_ids()
    index = 0
    true_predictions_indexes = []
    for image_id in train_image_ids:
        corrects = load_correct_predictions_of(image_id)
        for predict in corrects:
            if predict:
                true_predictions_indexes.append(index)
            index += 1
    return np.array(true_predictions_indexes)

def load_activations_of(image_id) -> np.ndarray:
    return np.load(activation_path(f'{image_id}.npz'))['dropouts']


def load_correct_predictions_of(image_id) -> np.ndarray:
    return np.load(prediction_path(f'{image_id}.npz'))['correct']
