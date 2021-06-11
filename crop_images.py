# adjusted from: https://github.com/M-Nauta/ProtoTree/tree/main/preprocess_data
from os.path import join
from PIL import Image
from tqdm import tqdm
from config import dataset
from utils.paths import ensure_directory_exists

path_images = dataset.path('images.txt')
images_save_path = dataset.path('images_cropped')
bbox_path = dataset.path('bounding_boxes.txt')

images = []
with open(path_images, 'r') as f:
  for line in f:
    images.append(list(line.strip('\n').split(',')))

bounding_boxes = dict()
with open(bbox_path, 'r') as bf:
  for line in bf:
    image_id, x, y, w, h = tuple(map(float, line.split(' ')))
    bounding_boxes[int(image_id)] = (x, y, w, h)

for k in tqdm(range(len(images))):
  image_id, path_from_images_root = images[k][0].split(' ')
  image_id = int(image_id)
  [source_folder_name, source_file_name] = path_from_images_root.split('/')
  target_folder = join(images_save_path, source_folder_name)
  source_image_path = join(dataset.path('images'), path_from_images_root)

  ensure_directory_exists(target_folder)

  img = Image.open(source_image_path).convert('RGB')
  x, y, w, h = bounding_boxes[image_id]
  cropped_img = img.crop((x, y, x + w, y + h))

  cropped_img.save(join(target_folder, source_file_name))
