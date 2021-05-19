from sklearn import tree
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import glob
from segmentize import _extract_patch, _return_superpixels
from PIL import Image

# X, y = 1, 2
# file_name = []
# y_train = []
# output = []
# fig, ax = plt.subplots(figsize=(10, 10))
# def similarity(a, b):
#   dist = np.linalg.norm(a - b)
#   return  dist
#
# with open("activation.csv", "r") as csvfile:
#   csvreader = csv.reader(csvfile)
#   for row in csvreader:
#     if not (row):
#       continue
#     file_name.append(row[1])
#     y_train.append(row[0])
#     output.append(row[2:])
#
# output = np.asarray(output, dtype=np.float64, order='C')
# centers = []
#
# with open("centers_100.csv", "r") as csvfile:
#   csvreader = csv.reader(csvfile)
#   for row in csvreader:
#     centers.append(row)
# centers = np.asarray(centers, dtype=np.float64, order='C')
# print(centers[0])
#
# X = []
# for o in output:
#   feas = []
#   for c in centers:
#     feas.append(similarity(o, c))
#   X.append(feas)
# X = np.array(X)
# print(X.shape)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, y_train)
# print('depth', clf.get_depth())
# print('leaves', clf.get_n_leaves())
# pred = clf.predict(X)
# print(accuracy_score(y_train, pred))
# print(clf.score(X, y_train, sample_weight=None))
# # tree.plot_tree(clf, ax=ax)
# # plt.savefig('tree.eps',format='eps',bbox_inches = "tight")
# # print("finish")
# # plt.show()
import torch
from torchvision import transforms

preprocess = transforms.Compose([
    #transforms.Resize((600, 600), Image.BILINEAR),
    # transforms.CenterCrop((448, 448)),
    # transforms.RandomHorizontalFlip(),  # only if train
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True, **{'topN': 6, 'device':'cpu', 'num_classes': 200})
model.eval()

path = 'data/CUB_200_2011/dataset/test_crop/001*'
index  = 0
shape = (448, 448)
image_activations= []
for folder in glob.glob(path):
  files = glob.glob(folder + '/*.jpg')[:10]
  for k in files:
    test_X = []
    image_activations = []
    img = Image.open(k)
    im2arr = np.array(img.resize(shape, Image.BILINEAR))
    # Normalize pixel values to between 0 and 1.
    im2arr = np.float32(im2arr) / 255.0
    image_superpixels, image_patches = _return_superpixels(im2arr)
    for image_superpixel in image_superpixels:
      input_tensor = preprocess(image_superpixel)
      input_batch = input_tensor.unsqueeze(0)
      top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(
        input_batch)
      for c in concat_logits:
        image_activations.append(c.detach().numpy())
    for o in image_activations:
      feas = []
      for c in centers:
        feas.append(similarity(o, c))
      test_X.append(feas)
    test_X = np.array(test_X)

    print(clf.predict(test_X))