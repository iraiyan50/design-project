
# CELL 1: Setup and Installation
!pip install opencv-python-headless
!pip install pillow
!pip install matplotlib


# CELL 2: Import Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
import io

# CELL 3: Define Model Architecture (PSPNet + ResNet)

# ResNet Building Blocks
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, deep_base=True):
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

# PPM Module
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

# PSPNet
class PSPNet(nn.Module):
    def __init__(self, num_classes, layers=50, bins=[1, 2, 3, 6],
                 bottleneck_dim=512, dropout=0.1, zoom_factor=8,
                 use_ppm=True, m_scale=False):
        super(PSPNet, self).__init__()
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.m_scale = m_scale
        self.bottleneck_dim = bottleneck_dim

        if layers == 50:
            resnet = resnet50()
        else:
            resnet = resnet101()

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu,
                                    resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        if self.m_scale:
            fea_dim = 1024 + 512
        else:
            fea_dim = 2048

        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
            self.bottleneck = nn.Sequential(
                nn.Conv2d(fea_dim, self.bottleneck_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
            )
        self.classifier = nn.Conv2d(self.bottleneck_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x_size = x.size()
        H = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        W = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x_2 = self.layer2(x)
        x_3 = self.layer3(x_2)

        if self.m_scale:
            x = torch.cat([x_2, x_3], dim=1)
        else:
            x = self.layer4(x_3)

        x = self.ppm(x)
        x = self.bottleneck(x)
        x = self.classifier(x)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

# ============================================================================
# CELL 4: Configuration - PASCAL VOC 2012 Split 3
# ============================================================================

# Model Configuration for Split 3
NUM_CLASSES = 16  # 15 base classes + 1 background
LAYERS = 50  # 50 or 101 depending on your ResNet backbone
IMAGE_SIZE = 473  # Input image size
BINS = [1, 2, 3, 6]
BOTTLENECK_DIM = 512
DROPOUT = 0.1
M_SCALE = False  # Set to True if you used m_scale in training

# PASCAL VOC Dataset normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# PASCAL VOC Split 3 - Base Classes (15 classes)
SPLIT_3_BASE_CLASSES = {
    0: 'background',
    1: 'airplane',      # Base
    2: 'bird',          # Base
    3: 'bottle',        # Base
    4: 'bus',           # Base
    5: 'cat',           # Base
    6: 'chair',         # Base
    7: 'cow',           # Base
    8: 'dining table',  # Base
    9: 'horse',         # Base
    10: 'person',       # Base
    11: 'potted plant', # Base
    12: 'sheep',        # Base
    13: 'sofa',         # Base
    14: 'train',        # Base
    15: 'tv'            # Base
}

# Novel classes in Split 3 (not in base model): bicycle, boat, car, dog, motorbike

# Color map with PASCAL VOC official colors for Split 3 base classes
def get_color_map_split3(num_classes=16):
    """PASCAL VOC color palette for Split 3 base classes"""
    colors = np.array([
        [0, 0, 0],       # 0: background (black)
        [128, 0, 0],     # 1: airplane (maroon)
        [128, 128, 0],   # 2: bird (olive)
        [128, 0, 128],   # 3: bottle (purple)
        [0, 128, 128],   # 4: bus (teal)
        [64, 0, 0],      # 5: cat (dark red)
        [192, 0, 0],     # 6: chair (red)
        [64, 128, 0],    # 7: cow (green)
        [192, 128, 0],   # 8: dining table (yellow-brown)
        [192, 0, 128],   # 9: horse (magenta)
        [192, 128, 128], # 10: person (pink)
        [0, 64, 0],      # 11: potted plant (dark green)
        [128, 64, 0],    # 12: sheep (brown)
        [0, 192, 0],     # 13: sofa (bright green)
        [128, 192, 0],   # 14: train (lime)
        [0, 64, 128],    # 15: tv (blue)
    ], dtype=np.uint8)
    return colors

COLOR_MAP = get_color_map_split3(NUM_CLASSES)

# Print configuration
print("=" * 70)
print("PASCAL VOC 2012 - Split 3 Configuration")
print("=" * 70)
print(f"Number of Classes: {NUM_CLASSES}")
print(f"ResNet Layers: {LAYERS}")
print(f"Image Size: {IMAGE_SIZE}")
print(f"\nBase Classes (15):")
for i in range(1, NUM_CLASSES):
    print(f"  Class {i}: {SPLIT_3_BASE_CLASSES[i]}")
print(f"\nNovel Classes (5 - NOT in base model):")
print("  bicycle, boat, car, dog, motorbike")
print("=" * 70)


