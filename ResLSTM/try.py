import natsort
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
device = torch.device('cpu')
import random
import glob  
import Model
import dataset

################################################################
#Getting Data ready 

seed_constant = 42
torch.manual_seed(seed_constant)
np.random.seed(seed_constant)
random.seed(seed_constant)

x = np.arange(1, 105)
np.random.shuffle(x)
videos_validation = x[:16]
videos_test = x[16: 16+22]
videos_train = x[16+22:]

filenames_train = []
labels_train = []
filenames_validation = []
labels_validation = []
filenames_test = []
labels_test = []

path_videos = '/home/kirtan/Documents/FYProject/archive/Videos/Videos/'
path_frames = '/home/kirtan/Documents/FYProject/archive/Videos/Frames/'

for vid in videos_train:
    folder = path_frames + "video{}/".format(vid)
    frames = glob.glob(folder + 'frame*.jpg')
    frames = natsort.natsorted(frames)
    filenames_train = np.append(filenames_train,frames)
    labels_path = path_frames + "video{}/".format(vid) + "labels{}.npy".format(vid)
    labels_array = np.load(labels_path)
    labels_list = list(labels_array)
    labels_train = np.append(labels_train,labels_list)

filenames_train = np.array(filenames_train)
labels_train = np.array(labels_train)

for vid in videos_test:
    folder = path_frames + "video{}/".format(vid)
    frames = glob.glob(folder + 'frame*.jpg')
    frames = natsort.natsorted(frames)
    filenames_test = np.append(filenames_test,frames)
    labels_path = path_frames + "video{}/".format(vid) + "labels{}.npy".format(vid)
    labels_array = np.load(labels_path)
    labels_list = list(labels_array)
    labels_list = np.asarray(labels_list).astype('float32').reshape((-1,1))
    labels_test = np.append(labels_test,labels_list)
    
filenames_test = np.array(filenames_test)
labels_test = np.array(labels_test)

for vid in videos_validation:
    folder = path_frames + "video{}/".format(vid)
    frames = glob.glob(folder + 'frame*.jpg')
    frames = natsort.natsorted(frames)
    filenames_validation = np.append(filenames_validation,frames)
#     filenames_validation.append(frames)
    labels_path = path_frames + "video{}/".format(vid) + "labels{}.npy".format(vid)
    labels_array = np.load(labels_path)
    labels_list = list(labels_array)
    labels_list = np.asarray(labels_list).astype('float32').reshape((-1,1))
    labels_validation = np.append(labels_validation,labels_list)

filenames_val = np.array(filenames_validation)
labels_val = np.array(labels_validation)

new_labels = labels_val + labels_train + labels_test
print(new_labels)