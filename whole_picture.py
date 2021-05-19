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
target_classes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']

for folder in glob.glob(path):
  target_class = folder.split('/')[-1].split('.')[0]
  if target_class not in target_classes:
    continue
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
      output.append(c.detach().numpy())
    ytarget.append(target_class)
print(ytarget)

with open("centers_1000.csv", "r") as csvfile:
  csvreader = csv.reader(csvfile)
  for row in csvreader:
    centers.append(row)
centers = np.asarray(centers, dtype=np.float64, order='C')
print(centers[0])

X = []
for o in output:
  feas = []
  for c in centers:
    feas.append(similarity(o, c))
  X.append(feas)
X = np.array(X)
print(X.shape)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, ytarget)
print('depth', clf.get_depth())
print('leaves', clf.get_n_leaves())
pred = clf.predict(X)
print(accuracy_score(ytarget, pred))
print(clf.score(X, ytarget, sample_weight=None))

print("--- %s seconds ---" % (time.time() - start_time))

# tree.plot_tree(clf, ax=ax)
# plt.savefig('tree.eps',format='eps',bbox_inches = "tight")
# print("finish")
# plt.show()


preprocess = transforms.Compose([
    #transforms.Resize((600, 600), Image.BILINEAR),
    # transforms.CenterCrop((448, 448)),
    # transforms.RandomHorizontalFlip(),  # only if train
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


path = 'data/CUB_200_2011/dataset/test_crop/*'
index = 0
shape = (448, 448)
image_activations= []
camp = []
y_test = []
test_X = []
for tfolder in glob.glob(path):
  test_files = glob.glob(tfolder + '/*.jpg')
  tc = tfolder.split('/')[-1].split('.')[0]
  if tc not in target_classes:
    continue
  for k in test_files:
    y_test.append(tc)
    img = Image.open(k)
    im2arr = np.array(img.resize(shape, Image.BILINEAR))
    # Normalize pixel values to between 0 and 1.
    im2arr = np.float32(im2arr) / 255.0
    input_tensor = preprocess(im2arr)
    input_batch = input_tensor.unsqueeze(0)
    top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(
      input_batch)
    for c in concat_logits:
      image_activations.append(c.detach().numpy())
    # image_superpixels, image_patches = _return_superpixels(im2arr)
    # for image_superpixel in image_superpixels:
    #   input_tensor = preprocess(image_superpixel)
    #   input_batch = input_tensor.unsqueeze(0)
    #   top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(
    #     input_batch)
    #   for c in concat_logits:
    #     image_activations.append(c.detach().numpy())

for o in image_activations:
  feas = []
  for c in centers:
    feas.append(similarity(o, c))
  test_X.append(feas)

final_res = clf.predict(test_X)
print(final_res, y_test)
print(accuracy_score(final_res, y_test))
print("--- %s seconds ---" % (time.time() - start_time))