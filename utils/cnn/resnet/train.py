# This file was provided

import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from net_fc import Net
import argparse

# VARIABLES
from utils.paths import data_path

num_features = 128 #or 256. Size of the latent embedding
path_to_saved_model = data_path(f'pre-trained-models/model_state_{num_features}.pth')
path_to_training_dataset = "" #make sure folder structure is as described here: https://pytorch.org/vision/stable/datasets.html#imagefolder
path_to_test_dataset = "" #make sure folder structure is as described here: https://pytorch.org/vision/stable/datasets.html#imagefolder
img_size = 448
batch_size = 32

# function to normalize test images (and augment training images if needed)
def get_birds(augment: bool, train_dir:str, test_dir:str, img_size = 448): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),
            ]),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes
    
    return trainset, testset, classes, shape

# Determine if GPU should be used
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Obtain the dataset
trainset, testset, classes, shape  = get_birds(False, path_to_training_dataset, path_to_test_dataset, img_size)
c, w, h = shape

# Obtain the dataloaders
trainloader = torch.utils.data.DataLoader(trainset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=cuda,
                                        num_workers = 0 #can be increased to speed up memory. Set to 0 when you get an BrokenPipeError
                                        )
testloader = torch.utils.data.DataLoader(testset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        pin_memory=cuda,
                                        num_workers = 0
                                        )
num_classes = len(classes)


args = argparse.Namespace()
args.net = 'resnet50_inat'
args.num_features = num_features
args.disable_pretrained=True

# Load the saved checkpoint
net = Net(3, num_classes, args)
net = net.to(device=device)
net.load_state_dict(torch.load(path_to_saved_model))
net = net.to(device=device)
# Make sure the model is in evaluation mode. Very important because the dropout layer will otherwise remove values from the latent embedding! Also use this when collecting the clusters
net.eval()


# Get a batch of training data to analyse its size
xs, ys = next(iter(trainloader))
xs, ys = xs.to(device), ys.to(device)
print("Batch input shape: "+str(xs[0,:,:,:].shape))
print("Batch output shape: "+str(net(xs).shape))
print("Batch output shape before FC layer: "+str(net.add_on_layers(net.features(xs)).shape))

# Show progress on progress bar
train_iter = tqdm(enumerate(trainloader),
                total=len(trainloader),
                ncols=0)
# Show progress on progress bar
test_iter = tqdm(enumerate(testloader),
                total=len(testloader),
                ncols=0)

# Iterate through the training set to collect the latent embeddings
with torch.no_grad():
    for i, (xs, ys) in train_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Use the model to get the latent embedding
        latent = net.add_on_layers(net.features(xs)) #latent embedding
        print("latent embeddings for this batch: ", latent.shape, latent)
        # TODO do your own thing here, like collecting all latent embeddings. 

# Iterate through the test set
with torch.no_grad():
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Use the model to classify this batch of input data
        out = net(xs) #prediction
        
        latent = net.add_on_layers(net.features(xs)) #latent embedding
        print("latent embeddings for this batch: ", latent.shape, latent)
        ys_pred = torch.argmax(out, dim=1) #class prediction
        print("predictions for this batch: ", ys_pred.shape, ys_pred)
        # TODO do your own thing here, like collecting all latent embeddings or calculating accuracy. 
        

