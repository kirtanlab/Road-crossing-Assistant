# import natsort
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from tqdm import tqdm
device = torch.device('cuda')
import random
import glob  
import Model
import dataset
print(torch.cuda.is_available())

mobilenet_v3_large = models.mobilenet_v3_large(weights='DEFAULT')
print(mobilenet_v3_large)
print('============================================================')
mobilenet_v3_large = nn.Sequential(*list(mobilenet_v3_large.children())[:-1])
print(mobilenet_v3_large)