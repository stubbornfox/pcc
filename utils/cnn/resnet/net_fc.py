import argparse
import torch.nn as nn
from utils.cnn.resnet.features.resnet_features import resnet18_features, \
    resnet34_features, resnet50_features, resnet50_features_inat, \
    resnet101_features, resnet152_features


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features}


"""
    Create network with pretrained features, 1x1 convolutional layer and fully-connected layer

"""
def get_network(num_in_channels: int, num_classes: int, args: argparse.Namespace):
    # Define a conv net for estimating the probabilities at each decision node
    features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)            
    features_name = str(features).upper()
    
    if features_name.startswith('VGG') or features_name.startswith('RES'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    elif features_name.startswith('DENSE'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
    else:
        raise Exception('other base base_architecture NOT implemented')
    
    add_on_layers = nn.Sequential(
        nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=args.num_features, kernel_size=1, bias=False),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Dropout(p=0.1),
        nn.Flatten(),
        nn.Sigmoid()
    )

    fc = nn.Linear(args.num_features,num_classes)
    return features, add_on_layers, fc

class Net(nn.Module):
    def __init__(self, num_in_channels: int, num_classes: int, args: argparse.Namespace):
        super(Net, self).__init__()
        self.features, self.add_on_layers, self.fc = get_network(num_in_channels, num_classes, args)

    def forward(self, x):
        x = self.features(x)
        x = self.add_on_layers(x)
        x = self.fc(x)
        return x



