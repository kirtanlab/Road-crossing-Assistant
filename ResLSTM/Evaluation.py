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


seed_constant = 42
torch.manual_seed(seed_constant)
np.random.seed(seed_constant)
random.seed(seed_constant)

x = np.arange(1, 105)
np.random.shuffle(x)

videos_test = x[16: 16+22]


filenames_test = []
labels_test = []

path_videos = '/home/kirtan/Documents/FYProject/archive/Videos/Videos/'
path_frames = '/home/kirtan/Documents/FYProject/archive/Videos/Frames/'

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

test_dataset = dataset.dataset(filenames_test, labels_test,mode='Test')

input_size_lstm = 960  # Size of ResNet output (num_classes for ResNet-50)
hidden_size_lstm = 128
num_classes_lstm = 2  # Number of classes in your classification task

# lstm_model = Model.LSTMModel(input_size_lstm, hidden_size_lstm, num_classes_lstm)
lstm_model = nn.LSTM(input_size_lstm,hidden_size_lstm,num_classes_lstm,batch_first=True).to(device)
Linear = nn.Linear(hidden_size_lstm,1).to(device) #output is 1

mobilenet_v3_large = models.mobilenet_v3_large(weights='DEFAULT')
mobilenet_v3_large = nn.Sequential(*list(mobilenet_v3_large.children())[:-1])
mobilenet_v3_large.eval().to(device)

checkpoint = torch.load("/home/kirtan/Documents/FYProject/Epoch_21_mobile_v3_v2.pt")
lstm_model.load_state_dict(checkpoint['net1'])
Linear.load_state_dict(checkpoint['net2'])
# resnet.load_state_dict(checkpoint['net3'])

lstm_model.eval()
Linear.eval()

val_all_true_labels = []
val_all_pred_labels = []
val_losses = []
val_accs = []
val_total = 0
val_correct = 0
val_accuracy =0.0
val_loss = 0.0
with tqdm(test_dataset, desc="Test", unit="batch", leave=False) as t:
    for batch in t:
        if len(batch['label']) == 320:
            if 'data' in batch:
                images = batch['data']
                labels = batch['label']
                with torch.no_grad():
                    features = mobilenet_v3_large(images)
                    features = features.to(device)

                batch_size, input_channels, height, width = features.shape
                input_size_resnet = input_channels * height * width
                sequence_length = 1
                features_reshaped = features.view(batch_size, sequence_length, input_size_resnet).to(device=device)

                lstm_output,_ = lstm_model(features_reshaped)
                lstm_output = lstm_output[:,-1,:]
                lstm_output = Linear(lstm_output)
                lstm_output = lstm_output.view(-1)

                val_loss_item = torch.nn.functional.binary_cross_entropy_with_logits(lstm_output.to(device), labels.float().to(device))
        
                val_loss = val_loss_item.item()
                # val_loss += add_regularization(lstm_model)

                predicted = (lstm_output >= 0.5).int()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_all_true_labels.extend(labels.tolist())
                val_all_pred_labels.extend(predicted.tolist())
                print(predicted.tolist())
                val_accuracy = 100 * val_correct / val_total
                print(val_accuracy,val_loss)


val_precision = precision_score(val_all_true_labels, val_all_pred_labels, average='macro')  
val_recall = recall_score(val_all_true_labels, val_all_pred_labels, average='macro')  

print(f"val_Precision: {val_precision:.3f}, val_Recall: {val_recall:.3f}")
print(f"Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.3f}")

# import natsort
# import torch
# import torch.nn as nn
# from torchvision import datasets, models, transforms
# from torchvision.models import resnet50, ResNet50_Weights
# import numpy as np
# from torch.utils.data import Dataset,DataLoader
# import os
# from PIL import Image
# import torch.optim as optim
# from datetime import datetime
# from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
# from tqdm import tqdm
# device = torch.device('cpu')
# import random
# import glob  
# import Model
# import dataset


# seed_constant = 42
# torch.manual_seed(seed_constant)
# np.random.seed(seed_constant)
# random.seed(seed_constant)

# x = np.arange(1, 105)
# np.random.shuffle(x)

# videos_test = x[16: 16+22]


# filenames_test = []
# labels_test = []

# path_videos = '/home/kirtan/Documents/FYProject/archive/Videos/Videos/'
# path_frames = '/home/kirtan/Documents/FYProject/archive/Videos/Frames/'

# for vid in videos_test:
#     folder = path_frames + "video{}/".format(vid)
#     frames = glob.glob(folder + 'frame*.jpg')
#     frames = natsort.natsorted(frames)
#     filenames_test = np.append(filenames_test,frames)
#     labels_path = path_frames + "video{}/".format(vid) + "labels{}.npy".format(vid)
#     labels_array = np.load(labels_path)
#     labels_list = list(labels_array)
#     labels_list = np.asarray(labels_list).astype('float32').reshape((-1,1))
#     labels_test = np.append(labels_test,labels_list)
    
# filenames_test = np.array(filenames_test)
# labels_test = np.array(labels_test)

# test_dataset = dataset.dataset_v2(filenames_test, labels_test,mode='Test')

# input_size_lstm = 32768  # Size of ResNet output (num_classes for ResNet-50)
# hidden_size_lstm = 128
# num_classes_lstm = 2  # Number of classes in your classification task

# # lstm_model = Model.LSTMModel(input_size_lstm, hidden_size_lstm, num_classes_lstm)
# lstm_model = nn.LSTM(input_size_lstm,hidden_size_lstm,num_classes_lstm,batch_first=True).to(device)
# Linear = nn.Linear(hidden_size_lstm,1).to(device) #output is 1

# resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
# resnet = nn.Sequential(*list(resnet.children())[:-2])
# resnet.eval().to(device)


# checkpoint = torch.load("/home/kirtan/Documents/FYProject/3_lstm.pt")
# lstm_model.load_state_dict(checkpoint['net1'])
# Linear.load_state_dict(checkpoint['net2'])
# # resnet.load_state_dict(checkpoint['net3'])

# lstm_model.eval()
# Linear.eval()

# val_all_true_labels = []
# val_all_pred_labels = []
# val_losses = []
# val_accs = []
# val_total = 0
# val_correct = 0
# val_accuracy =0.0
# val_loss = 0.0
# with tqdm(test_dataset, desc="Test", unit="batch", leave=False) as t:
#     for batch in t:
#         if len(batch['label']) == 80:
#             if 'data' in batch:
#                 images = batch['data']
#                 labels = batch['label']
#                 labels = labels.view(8, 10, 1)
#                 with torch.no_grad():
#                     features = resnet(images)
#                     features = features.to(device)

#                 batch_size, input_channels, height, width = features.shape
#                 input_size_resnet = input_channels * height * width
#                 sequence_length = 10
#                 batch_size = 8
#                 features_reshaped = features.view(batch_size, sequence_length, input_size_resnet).to(device=device)

#                 lstm_output,_ = lstm_model(features_reshaped)
#                 # lstm_output = lstm_output[:,-1,:]
#                 lstm_output = Linear(lstm_output)
#                 # lstm_output = lstm_output.view(-1)

#                 val_loss_item = torch.nn.functional.binary_cross_entropy_with_logits(lstm_output.to(device), labels.float().to(device))
        
#                 val_losses.append(val_loss)

#                 lstm_output = lstm_output.view(80)
#                 labels = labels.view(80)
#                 predicted = (lstm_output >= 0.6).int()

#                 val_total = labels.size(0)
#                 val_correct = (predicted == labels).sum().item()

#                 val_all_true_labels.extend(labels.tolist())
#                 val_all_pred_labels.extend(predicted.tolist())
#                 print(predicted.tolist())
#                 val_accuracy = 100 * val_correct / val_total
#                 print(val_accuracy,val_loss)


# val_precision = precision_score(val_all_true_labels, val_all_pred_labels, average='macro')  
# val_recall = recall_score(val_all_true_labels, val_all_pred_labels, average='macro')  

# print(f"val_Precision: {val_precision:.3f}, val_Recall: {val_recall:.3f}")
# print(f"Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.3f}")