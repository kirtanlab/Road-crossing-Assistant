import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset,Dataset
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import random
device = torch.device('cpu')
print(device)
import glob  
import natsort
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import timm

seed_constant = 42
torch.manual_seed(seed_constant)
np.random.seed(seed_constant)
random.seed(seed_constant)

path_videos = '/home/kirtan/Documents/FYProject/archive/Videos/Videos/'
path_frames = '/home/kirtan/Documents/FYProject/archive/Videos/Frames/'
# checkpoint_path = "./checkpoints/"

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

# for vid in videos_test:
#     folder = path_frames + "video{}/".format(vid)
#     frames = glob.glob(folder + 'frasme*.jpg')
#     frames = natsort.natsorted(frames)
#     filenames_test = np.append(filenames_test,frames)
#     labels_path = path_frames + "video{}/".format(vid) + "labels{}.npy".format(vid)
#     labels_array = np.load(labels_path)
#     labels_list = list(labels_array)
#     labels_list = np.asarray(labels_list).astype('float32').reshape((-1,1))
#     labels_test = np.append(labels_test,labels_list)
    
# filenames_test = np.array(filenames_test)
# labels_validation = np.array(labels_validation)

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

filenames_validation = np.array(filenames_validation)
labels_validation = np.array(labels_validation)

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels,transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path)

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(label,dtype=torch.int64)
        # print('len(image_path),len(label)',image_path.size(),label.size())
        return {"pixel_value": image, "label": label}

resize_size = 232
crop_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
image_transform = transforms.Compose([
    Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
    CenterCrop(crop_size),
    ToTensor(),
    Normalize(mean=mean, std=std)
])
# transform_aug = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     # transforms.RandomResizedCrop(size=(8, 8), scale=(0.8, 1.0)),
#     transforms.ToTensor(),
# ])


train_dataset = CustomDataset(filenames_train, labels_train, transform=image_transform)
# train_dataset_aug = CustomDataset(filenames_train, labels_train, transform=transform_aug)
test_dataset = CustomDataset(filenames_test, labels_test, transform=image_transform)
val_dataset = CustomDataset(filenames_validation, labels_validation, transform=image_transform)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False,drop_last=True,num_workers=6)
# train_dataloader_aug = DataLoader(train_dataset_aug, batch_size=16, shuffle=False,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False,drop_last=True,num_workers=6)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False,drop_last=True,num_workers=6)

# print(labels)
model = timm.create_model("convnext_base_in22ft1k", pretrained=True) # Will take a moment
class ReshapeLayer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
model.head = head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),   # Change the pool_type if needed
    ReshapeLayer(),
    nn.LayerNorm((1024,),eps=1e-06,elementwise_affine=True),     # Assuming you want to keep LayerNorm
    nn.Flatten(start_dim=1,end_dim=-1),
    nn.Dropout(0.0,inplace=False),        # Adjust dropout if needed
    nn.Linear(1024, 1,bias=True)      # Change the number of output features to 2
)


from tqdm import tqdm 

# L2 regularization
l2_lambda = 0.01  # L2 regularization strength
best_val_loss = float('inf')
best_model_state_dict = None
optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
# Regularization
def add_regularization(model, lambda_reg):
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)  # L2 norm
    return l2_lambda * l2_reg

pos_weight = torch.tensor([1.92]).to(device)                    

model = model.to(device)
model.train()
threshold = 0.6

for epoch in range(100):
    # print("Epoch:", epoch)
    correct = 0
    total = 0
    running_loss = 0.0
    aug_correct = 0
    aug_total = 0
    aug_running_loss = 0.0
    with tqdm(train_dataloader, desc=f"Epoch {epoch + 1} (Training)", unit="batch", leave=False) as t:
        for idx, batch in enumerate(tqdm(train_dataloader,desc=f"Epoch {epoch + 1}", unit="batch", leave=False)):
            # Move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch["pixel_value"])
            outputs = outputs.squeeze()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs,batch["label"].float(),pos_weight=pos_weight)
            loss += add_regularization(model, l2_lambda)  # Apply L2 regularization
            loss.backward()
            optimizer.step()

            # Metrics
            total += batch["label"].shape[0]
            sigmoid = torch.sigmoid(outputs)
            predicted = (sigmoid > threshold).int()

            correct += (predicted == batch["label"]).sum().item()
            running_loss += loss.item()

            accuracy = correct / total
            avg_loss = running_loss / (idx + 1)
            t.set_postfix(loss=avg_loss, train_accuracy=f"{accuracy:.2f}%")

    # print("\nTrain data\n")
    # print(f"Loss steps:", avg_loss)
    # print(f"Accuracy steps:", accuracy)
            
     # Training loop with augmented data
    with tqdm(train_dataloader_aug, desc=f"Epoch {epoch + 1} (Augmentation)", unit="batch", leave=False) as t:
        for idx, batch in enumerate(tqdm(train_dataloader_aug,desc=f"Epoch {epoch + 1}", unit="batch", leave=False)):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch["pixel_value"])
            outputs = outputs.squeeze()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs,batch["label"].float(),pos_weight=pos_weight)
            loss += add_regularization(model, l2_lambda)  # Apply L2 regularization
            loss.backward()
            optimizer.step()

            aug_total += batch["label"].shape[0]
            sigmoid = torch.sigmoid(outputs)
            predicted = (sigmoid > threshold).int()

            aug_correct += (predicted == batch["label"]).sum().item()
            aug_running_loss += loss.item() 
            
            accuracy = correct / total
            avg_loss = aug_running_loss / (idx + 1)

            t.set_postfix(loss=avg_loss, train_accuracy=f"{accuracy:.2f}%")
    
        
    # print("\nAug data\n")
    # print(f"Loss (Augmented Data):", avg_loss)
    # print(f"Accuracy (Augmented Data):", accuracy)

    # Validation loop
    model.eval()
    val_correct = 0
    val_total = 0
    val_running_loss = 0.0
    val_predictions = []
    val_labels = []
    
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader,desc=f"Epoch {epoch + 1}", unit="batch", leave=False):
            if len(val_batch) == 2: 
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                
                outputs = model(val_batch["pixel_value"])
                outputs = outputs.squeeze()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs,val_batch["label"].float(),pos_weight=pos_weight)

                val_total += val_batch["label"].shape[0]
                sigmoid = torch.sigmoid(outputs)
                val_predicted = (sigmoid > threshold).int()

                print("val_predicted", val_predicted)
                val_correct += (val_predicted == val_batch["label"]).sum().item()
                val_running_loss += loss.item()

                val_predictions.extend(val_predicted.tolist())
                val_labels.extend(val_batch["label"].tolist())
            else: 
                continue
    val_accuracy = val_correct / val_total
    val_avg_loss = val_running_loss / len(val_dataloader)
    print("\Val data\n")
    print("Validation Loss:", val_avg_loss)
    print("Validation Accuracy:", val_accuracy)
    
    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        best_model_state_dict = model.state_dict()
        torch.save(best_model_state_dict, "conv_v1.pt")
    torch.save( model.state_dict(), f"ConvNext_{epoch+1}_best.pt")
    # Calculate precision, recall, and precision at recall
    precision = precision_score(val_labels, val_predictions)
    recall = recall_score(val_labels, val_predictions)
    precision_at_recall = precision_recall_curve(val_labels, val_predictions, pos_label=1)

    print("Precision:", precision)
    print("Recall:", recall)
    print("Precision at Recall:", precision_at_recall)

    model.train()

