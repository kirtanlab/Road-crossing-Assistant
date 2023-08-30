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
print(torch.cuda.is_available())
from transformers import ViTImageProcessor, ViTModel
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


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.15, contrast=(0.8, 1.5), saturation=(0.6, 3))
    ], p=0.5)
])
transform_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomResizedCrop(size=(8, 8), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])


train_dataset = dataset.CustomDataset(filenames_train, labels_train, transform=image_transform)
train_dataset_aug = dataset.CustomDataset(filenames_train, labels_train, transform=transform_aug)
# print(type(train_dataset_aug[0]['pixel_value']))
# print(type(train_dataset_aug[0]['label']))
# print(train_dataset_aug[0]['pixel_value'])
test_dataset = dataset.CustomDataset(filenames_test, labels_test, transform=image_transform)
val_dataset = dataset.CustomDataset(filenames_validation, labels_validation, transform=image_transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_dataloader_aug = DataLoader(train_dataset_aug, batch_size=64, shuffle=True)
# batch = next(iter(train_dataloader_aug))
# print(batch)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

################################################################

################################################################

model = models.swin_v2_t()

in_features = model.head.in_features
model.head = nn.Sequential(
    nn.Linear(in_features, 1), 
    nn.Sigmoid()  
)

best_f1score = 0.0
best_model_state_dict = None
l2_lambda = 0.01
optimizer = optim.Adam([{'params':model.parameters()}], lr=0.001)

def add_regularization(model):
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)  # L2 norm
    return l2_lambda * l2_reg


num_epochs = 100
for epoch in range(num_epochs):
    train_all_true_labels = []
    train_all_pred_labels = []
    val_all_true_labels = []
    val_all_pred_labels = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    #Training 
    # Use tqdm for a progress bar over the DataLoader
    with tqdm(train_dataset, desc=f"Epoch {epoch + 1}", unit="batch", leave=False) as t:
        for batch in t:
            if 'data' in batch:
                images = batch['data']
                labels = batch['label']
                model.train()
                # Linear.train()
                # lstm_model.train()
                optimizer.zero_grad()

                # with torch.no_grad():
                # features = processor(images, return_tensors="pt")
                print(images.shape)
                lstm_output = model(images)
                print(lstm_output.shape)
                # features = features.last_hidden_state
                # print(features.shape)
                # batch_size = 8
                # sequence_length = 10
                # # print(features.shape)
                # features_reshaped = features.view(batch_size, -1).to(device)
                # # print(features_reshaped.shape)
                # lstm_output,_ = lstm_model(features_reshaped)

                # lstm_output = Linear(features_reshaped).to(device)
                pos_weight = torch.tensor([1.92]).to(device)
                
                # print('lstm_output: ', lstm_output.shape)
                train_loss = torch.nn.functional.binary_cross_entropy_with_logits(lstm_output.to(device), labels.float().to(device),pos_weight=pos_weight)
                train_losses.append(train_loss.to(device))
                
                train_loss += add_regularization(model)
                train_loss.backward()
                optimizer.step()

                lstm_output = lstm_output.view(80)
                labels = labels.view(80)
                predicted = (lstm_output >= 0.6).int()

                train_total = labels.size(0)
                train_correct = (predicted == labels).sum().item()

                train_all_true_labels.extend(labels.tolist())
                train_all_pred_labels.extend(predicted.tolist())
                
                train_accuracy = 100 * train_correct / train_total
                train_accs.append(train_accuracy)
                t.set_postfix(loss=train_loss.item(), train_accuracy=f"{train_accuracy:.2f}%")
                   
    
    #validation
    with tqdm(val_dataset, desc=f"Epoch {epoch + 1} (Validation)", unit="batch", leave=False) as t:
        for batch in t:
            if len(batch['label']) == 80:
                if 'data' in batch:
                    images = batch['data']
                    labels = batch['label'].to(device)
                    labels = labels.view(8, 10, 1)
                    lstm_model.eval()
                    Linear.eval()
                    model.eval()

                    features = model(images)
                    features = features.last_hidden_state
                    batch_size = 8
                    sequence_length = 10
                    
                    features_reshaped = features.view(batch_size, sequence_length, -1).to(device)

                    lstm_output,_ = lstm_model(features_reshaped)
                    lstm_output = Linear(lstm_output).to(device)
                    print(lstm_output)
                    val_loss = torch.nn.functional.binary_cross_entropy_with_logits(lstm_output.to(device), labels.float().to(device)).item()
                    val_losses.append(val_loss)

                    lstm_output = lstm_output.view(80)
                    labels = labels.view(80)
                    predicted = (lstm_output >= 0.6).int()

                    val_total = labels.size(0)
                    val_correct = (predicted == labels).sum().item()

                    val_all_true_labels.extend(labels.tolist())
                    val_all_pred_labels.extend(predicted.tolist())

                    val_accuracy = 100 * val_correct / val_total
                    val_accs.append(val_accuracy)
                    t.set_postfix(val_accuracy=f"{val_accuracy:.2f}",val_loss=f"{val_loss:.3f}")
                 
                    

    train_precision = precision_score(train_all_true_labels, train_all_pred_labels, average='macro')
    train_recall = recall_score(train_all_true_labels, train_all_pred_labels, average='macro')
    
    val_precision = precision_score(val_all_true_labels, val_all_pred_labels, average='macro') 
    val_recall = recall_score(val_all_true_labels, val_all_pred_labels, average='macro') 

    train_accuracy = sum(train_accs) / len(train_accs)
    val_accuracy = sum(val_accs) / len(val_accs)

    train_loss = sum(train_losses) / len(train_losses)
    val_loss = sum(val_losses) / len(val_losses)

    train_f1score = 2 * (train_precision * train_recall) / (train_precision + train_recall)
    val_f1score = 2 * (val_precision * val_recall) / (val_precision + val_recall)

    if val_f1score > best_f1score:
        print("Best precision or recall")
        best_f1score = val_f1score
        # best_model_state_dict = lstm_model.state_dict()
        state = {'net1':lstm_model.state_dict(),
                'net2':Linear.state_dict(),
                'optimizer':optimizer.state_dict(),
                'model': model.state_dict()}
        torch.save(state, "Vit_1_best.pt")
    state = {'net1':lstm_model.state_dict(),
                'net2':Linear.state_dict(),
                'optimizer':optimizer.state_dict(),
                'model': model.state_dict()}
    torch.save(state, f"Vit_{epoch+1}_best.pt")
    print(f"Epoch {epoch + 1}, train_Precision: {train_precision:.3f}, train_Recall: {train_recall:.3f}, train_f1score: {train_f1score:.3f},val_Precision: {val_precision:.3f}, val_Recall: {val_recall:.3f},val_f1score: {val_f1score:.3f}")

    print(f"Epoch {epoch + 1}, Final Training Accuracy: {train_accuracy:.2f}%, Training_Loss: {train_loss: .3f}")
    print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.3f}")

################################################################

##her f1score is 0.83