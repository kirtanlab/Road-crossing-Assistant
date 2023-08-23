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

# image_transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.RandomApply([
#         transforms.ColorJitter(brightness=0.15, contrast=(0.8, 1.5), saturation=(0.6, 3))
#     ], p=0.5)
# ])
# transform_aug = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     # transforms.RandomResizedCrop(size=(8, 8), scale=(0.8, 1.0)),
#     transforms.ToTensor(),
# ])

################################################################

################################################################
#Custom Dataset Class

train_dataset = dataset.dataset(filenames_train, labels_train,mode='Train')
# img_dataset = dataset.dataset(filenames_train, labels_train,mode='Train',work='FE')
# train_dataset_aug = dataset.dataset(filenames_train, labels_train)
# print('train',train_dataset[4]['data']) #80 #almost same values except few
# print('label',train_dataset[4]['label'][0]) #80  
# print('len',train_dataset.__len__())
test_dataset = dataset.dataset(filenames_test, labels_test,mode='Test')
val_dataset = dataset.dataset(filenames_validation, labels_validation,mode="val")

# batch = next(iter(img_dataset))
# print(batch) #images

################################################################

################################################################
#Feature Extraction using ResNet & Passing Into LSTM
#Training 


resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-2])  # Exclude last two layers

resnet.eval()

input_size_lstm = 8192  # Size of ResNet output (num_classes for ResNet-50)
hidden_size_lstm = 128
num_classes_lstm = 2  # Number of classes in your classification task

lstm_model = Model.LSTMModel(input_size_lstm, hidden_size_lstm, num_classes_lstm)

criterion = torch.nn.CrossEntropyLoss()
best_val_loss = float('inf')
l2_lambda = 0.01
optimizer = optim.AdamW(lstm_model.parameters(), lr=0.005)
def add_regularization(model):
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)  # L2 norm
    return l2_lambda * l2_reg


num_epochs = 100

for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    val_correct = 0
    val_total = 0
    train_accuracy = 0
    val_accuracy = 0
    train_all_true_labels = []
    train_all_pred_labels = []
    val_all_true_labels = []
    val_all_pred_labels = []
    train_loss = 0
    val_loss = 0
    
    #Training 
    lstm_model.train()
    # Use tqdm for a progress bar over the DataLoader
    with tqdm(train_dataset, desc=f"Epoch {epoch + 1}", unit="batch", leave=False) as t:
        for batch in t:
            if len(batch['label']) == 80:
                if 'data' in batch:
                    images = batch['data']
                    labels = batch['label']

                    with torch.no_grad():
                        # resnet_without_last_layer = nn.Sequential(*list(resnet.children())[:-2])
                        features = resnet(images)
                        # print(features.shape)
                    # print(features.shape) 
                    batch_size, input_channels, height, width = features.shape
                    input_size_resnet = input_channels * height * width

                    # batch_size, input_size_resnet = features.shape
                    sequence_length = 1
                    features_reshaped = features.view(batch_size, sequence_length, input_size_resnet)

                    optimizer.zero_grad()

                    lstm_output = lstm_model(features_reshaped)

                    train_loss = criterion(lstm_output, labels)

                    optimizer.zero_grad()
                    train_loss += add_regularization(lstm_model)
                    train_loss.backward()
                    optimizer.step()

                    _, predicted = torch.max(lstm_output, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    train_all_true_labels.extend(labels.tolist())
                    train_all_pred_labels.extend(predicted.tolist())
                    
                    train_accuracy = 100 * train_correct / train_total
                    t.set_postfix(loss=train_loss.item(), train_accuracy=f"{train_accuracy:.2f}%", val_accuracy=f"{val_accuracy:.2f}%",val_loss=f"{val_loss:.5f}" )
    
    #validation
    lstm_model.eval()
    with tqdm(val_dataset, desc=f"Epoch {epoch + 1} (Validation)", unit="batch", leave=False) as t:
        for batch in t:
            if len(batch['label']) == 80:
                if 'data' in batch:
                    images = batch['data']
                    labels = batch['label']

                    with torch.no_grad():
                        features = resnet(images)

                    batch_size, input_channels, height, width = features.shape
                    input_size_resnet = input_channels * height * width
                    sequence_length = 1
                    features_reshaped = features.view(batch_size, sequence_length, input_size_resnet)

                    lstm_output = lstm_model(features_reshaped)

                    val_loss_item = criterion(lstm_output, labels)
                    val_loss = val_loss_item.item()
                    val_loss += add_regularization(lstm_model)

                    _, predicted = torch.max(lstm_output, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    val_all_true_labels.extend(labels.tolist())
                    val_all_pred_labels.extend(predicted.tolist())

                    val_accuracy = 100 * val_correct / val_total
                    t.set_postfix(loss=train_loss.item(), train_accuracy=f"{train_accuracy:.2f}%", val_accuracy=f"{val_accuracy:.2f}",val_loss=f"{val_loss:.3f}")
    

    train_precision = precision_score(train_all_true_labels, train_all_pred_labels, average='macro')  # Modify as needed
    train_recall = recall_score(train_all_true_labels, train_all_pred_labels, average='macro')  # Modify as needed
    
    val_precision = precision_score(val_all_true_labels, val_all_pred_labels, average='macro')  # Modify as needed
    val_recall = recall_score(val_all_true_labels, val_all_pred_labels, average='macro')  # Modify as needed

    print(f"Epoch {epoch + 1}, train_Precision: {train_precision:.3f}, train_Recall: {train_recall:.3f}, val_Precision: {val_precision:.3f}, val_Recall: {val_recall:.3f}")

    print(f"Epoch {epoch + 1}, Final Training Accuracy: {train_accuracy:.2f}%, Training_Loss: {train_loss: .3f}")
    print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.3f}")




# with torch.no_grad():
#     output = resnet(batch)

# print(output.shape)
# print(batch.shape)



# batch_size, input_size_resnet = output.shape

# sequence_length = 1

# output_reshaped = output.view(batch_size, sequence_length, input_size_resnet)

# lstm_output = lstm_model(output_reshaped)

# print(lstm_output.shape)  


################################################################

