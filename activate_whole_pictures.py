from sklearn import tree
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import glob
from segmentize import _extract_patch, _return_superpixels
from PIL import Image
import torch
from torchvision import transforms

X, y = 1, 2
file_name = []
y_train = []
output = []
# fig, ax = plt.subplots(figsize=(10, 10))
def similarity(a, b):
  dist = np.linalg.norm(a - b)
  return  dist
import time
start_time = time.time()
preprocess = transforms.Compose([
    #transforms.Resize((600, 600), Image.BILINEAR),
    # transforms.CenterCrop((448, 448)),
    # transforms.RandomHorizontalFlip(),  # only if train
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True, **{'topN': 6, 'device':'cpu', 'num_classes': 200})
model.eval()

path = 'data/CUB_200_2011/dataset/train_crop/*'
index = 0
shape = (448, 448)
centers = []
ytarget = []
count = 0
with open('whole_training_activation.csv', 'w', newline='') as csvfile:
  spamwriter = csv.writer(csvfile, delimiter=',')
  for folder in glob.glob(path):
    target_class = folder.split('/')[-1].split('.')[0]
    files = glob.glob(folder + '/*.jpg')
    for k in files:
      test_X = []
      image_activations = []
      img = Image.open(k)
      im2arr = np.array(img.resize(shape, Image.BILINEAR))
      # Normalize pixel values to between 0 and 1.
      im2arr = np.float32(im2arr) / 255.0
      input_tensor = preprocess(im2arr)
      input_batch = input_tensor.unsqueeze(0)
      top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(
        input_batch)
      for c in concat_logits:
        b = [target_class, k]
        b.extend(c.detach().numpy())
        spamwriter.writerow(b)
    count += 1
    print("{}/200".format(count))


