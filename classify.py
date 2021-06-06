from argparse import ArgumentParser
from os.path import exists, isfile

from PIL.Image import Image, open


def _open_image() -> tuple[str, Image]:
    parser = ArgumentParser()
    parser.add_argument(
        'image',
        help='Path to the image that should be classified'
    )
    arguments = parser.parse_args()
    path_to_image = arguments.image

    if not exists(path_to_image):
        print(f'"{path_to_image}" could not be found! Make sure this path points to a file on the filesystem!')
        exit(-1)

    if not isfile(path_to_image):
        print(f'"{path_to_image}" exists, but does not point to a file!')
        exit(-1)

    image = open(path_to_image)
    if not image.verify():
        print(f'"{path_to_image}" exists, but the file does not seem to be an image (according to PIL)!')
        exit(-1)

    return path_to_image, image

if __name__ == '__main__':
    path_to_image, image = _open_image()
