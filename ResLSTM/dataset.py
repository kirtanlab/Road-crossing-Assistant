import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
device = torch.device('cpu')

class dataset(Dataset):
    def __init__(self, image_paths, labels,mode,work=''):
        self.image_paths = image_paths
        self.labels = labels
        self.mode = mode
        self.work = work
        
    def __len__(self):
        return (len(self.image_paths)//320)

    def preprocess_function(self,filename,mode):
        if filename != "":
            image = Image.open(filename)
        else:
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # image = transforms.Resize((232, 232))(image)
        # image = transforms.ToTensor()(image)
        # image = image.to(torch.float32)
        # # print('image: ',image.shape)
        # image = transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )(image)
        
        composed_transform = transforms.Compose([
            transforms.Resize(232, interpolation=Image.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image=  composed_transform(image)
        return image
    
    def __getitem__(self, idx):
        
        # total 80 consicutive frames
        lower_value = idx * 320
        upper_value = idx * 320 + 319
        if(self.image_paths[upper_value+1]):

            filenames_range = self.image_paths[lower_value:upper_value+1]  #list of frames, +1 since slice excludes uppervalue 
            label_range = self.labels[lower_value:upper_value+1] # constant value of label 
            

            processed_images = [self.preprocess_function(image,self.mode) for image in filenames_range]

            processed_images = torch.stack(processed_images)

            labels = torch.stack([torch.tensor(label, dtype=torch.long) for label in label_range])
        
            return {'data': processed_images, 'label': labels}
        else:
            return {'data': None, 'label': None}
    
class dataset_v2(Dataset):
    def __init__(self, image_paths, labels,mode,work=''):
        self.image_paths = image_paths
        self.labels = labels
        self.mode = mode
        self.work = work
        
    def __len__(self):
        return (len(self.image_paths)//80)

    def preprocess_function(self,filename,mode):
        if filename != "":
            image = Image.open(filename)
        else:
            image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        
        image = transforms.Resize((128, 128))(image)
        image = transforms.ToTensor()(image)
        image = image.to(torch.float32)
        # print('image: ',image.shape)
        image = transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)
        if mode != 'Test':
            composed_transform = transforms.Compose([
                transforms.Resize(232, interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image=  composed_transform(image)
        return image
    
    def __getitem__(self, idx):
        
        # total 80 consicutive frames
        lower_value = idx * 80
        upper_value = idx * 80 + 79
        if(self.image_paths[upper_value+1]):

            filenames_range = self.image_paths[lower_value:upper_value+1]  #list of frames, +1 since slice excludes uppervalue 
            label_range = self.labels[lower_value:upper_value+1] # constant value of label 
            

            processed_images = [self.preprocess_function(image,self.mode) for image in filenames_range]

            processed_images = torch.stack(processed_images)

            labels = torch.stack([torch.tensor(label, dtype=torch.long) for label in label_range])
        
            return {'data': processed_images, 'label': labels}
        else:
            return {'data': None, 'label': None}

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
        return {"data": image, "label": label}
    




# class ViTDataset(Dataset):
#     def __init__(self, image_paths, labels,mode,work=''):
#         self.image_paths = image_paths
#         self.labels = labels
#         self.mode = mode
#         self.work = work
        
#     def __len__(self):
#         return (len(self.image_paths)//80)

#     def preprocess_function(self,filename,mode):
#         if filename != "":
#             image = Image.open(filename)
#         else:
#             image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        
#         image = transforms.Resize((224, 224))(image)
#         image = transforms.ToTensor()(image)
#         image = image.to(torch.float32)
#         # print('image: ',image.shape)
#         image = transforms.Normalize(
#            mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )(image)
        
#         if mode != 'Test':
#             image = transforms.RandomHorizontalFlip()(image)
#             image = transforms.RandomRotation(90)(image)
#             image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(image)
#             image = transforms.Normalize(mean=[0.5], std=[0.5])(image)
    
#         return image
    
#     def __getitem__(self, idx):

#         image_path = self.image_paths[idx]
#         label = self.labels[idx]

#         image = Image.open(image_path)

#         # Apply transformations
#         if self.transform is not None:
#             image = self.transform(image)

#         label = torch.tensor(label,dtype=torch.int64)
#         # print('len(image_path),len(label)',image_path.size(),label.size())
#         return {"data": image, "label": label}

  
    
