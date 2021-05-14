import os.path
import pickle

import torch
from skimage.segmentation import slic
from PIL import Image
import numpy as np
from glob import glob
from torchvision.transforms import transforms
from tqdm import tqdm
from Segment import Segment

def _extract_segments(image, mask, average_image_value=117):
    mask_expanded = np.expand_dims(mask, -1)
    patch = (mask_expanded * image + (
      1 - mask_expanded) * float(average_image_value) / 255)
    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
    image_resized = image.resize((224, 224), Image.BICUBIC)
    # np.array(image.resize((299, 299), Image.BICUBIC)).astype(float) / 255
    # plt.imshow(image_resized)
    # plt.show()
    return image_resized, patch

def _return_superpixels(im2arr):
    n_segmentss = [15, 50, 80]
    n_params = len(n_segmentss)
    unique_masks = []
    for i in range(n_params):
        param_masks = []
        segments_slic = slic(im2arr, n_segments=n_segmentss[i], compactness=20, sigma=1, start_label=1)
        for s in range(segments_slic.max()):
            mask = (segments_slic == s).astype(float)
            if np.mean(mask) > 0.001:
                unique = True
                for seen_mask in unique_masks:
                    jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                    if jaccard > 0.5:
                        unique = False
                        break
                if unique:
                    param_masks.append(mask)
        unique_masks.extend(param_masks)
        superpixels, segments = [], []
        while unique_masks:
            superpixel, segment = _extract_segments(im2arr, unique_masks.pop())
            superpixels.append(superpixel)
            segments.append(segment)

        return superpixels, segments

def _images_in_folder(folder_glob):
    """Iteration logic to load up each image from the birds"""
    outer_progress = tqdm(glob(folder_glob), position=0, leave=False)

    for folder_path in outer_progress:
        # Folder names are of the form "001.Black_footed_Albatross"
        folder_name = folder_path.split("/")[-1]
        bird_id, bird_name = folder_name.split('.')
        images = glob(folder_path + '/*.jpg')

        for image_file_path in tqdm(images, desc=bird_name, position=1, leave=False):
            outer_progress.refresh()
            # File names are of the form "Black_Footed_Albatross_0001_796111.jpg"
            file_name = image_file_path.split('/')[-1]
            suffix = file_name.upper().replace(bird_name.upper() + '_', '')
            image_id = int(suffix.split('_')[0])
            yield int(bird_id), image_id, Image.open(image_file_path)

def segment_source_images(
    source_image_directory='data/CUB_200_2011/dataset/train_crop/*',
    force_recreation=False,
    checkpoint_file='data/checkpoints/segments.obj',
) -> list[Segment]:
    """
    :param source_image_directory: Optional path to the cropped image folders
    :param force_recreation: recreate the checkpoint file instead of reusing it
    :param checkpoint_file: path to the checkpoint file
    :return: segments with attached metadata e.g. feature vector obtained from googlenet
    """
    if not force_recreation and os.path.isfile(checkpoint_file):
        print(f'Reading segments from cache ({checkpoint_file})...')
        with open(checkpoint_file, 'rb') as checkpoint:
            return pickle.load(checkpoint)
    else:
        print('Creating image segments...')

    segments = []
    model = torch.hub.load('pytorch/vision:v0.9.0', 'googlenet', pretrained=True)
    model.eval()
    preprocessing_pipeline = transforms.Compose([
        transforms.ToTensor(),
        # Values obtained from Prototree paper
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for bird_id, image_id, image in _images_in_folder(source_image_directory):
        im2arr = np.array(image.resize((224, 224), Image.BILINEAR))
        # Normalize pixel values to between 0 and 1.
        im2arr = np.float32(im2arr) / 255.0
        superpixels, _ = _return_superpixels(im2arr)
        for i, segment in enumerate(superpixels):
            segment_data = np.array(segment)
            segment_tensor = preprocessing_pipeline(segment)
            with torch.no_grad():
                mini_batch = segment_tensor.unsqueeze(0)
                features = model(mini_batch)

            segments.append(Segment(
                class_id=bird_id,
                image_id=image_id,
                segment_id=i,
                raw=segment_data,
                features=features
            ))

    print("Writing checkpoint file...")
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'wb') as checkpoint:
        pickle.dump(segments, checkpoint)

    return segments
