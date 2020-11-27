import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# 4. Input pipeline

# download to CIFAR10 dataset
# train_dataset = torchvision.datasets.CIFAR10(root='D:/pytorch_study/pytorch_tutorial/dataset', 
#                                             train = True, transform=transforms.ToTensor(), 
#                                             download = True)

image, label = train_dataset[0]
print(image.size())
print(label)

# data loader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=64, shuffle = True)
data_iter = iter(train_loader)

# Mini-batch images and labels
images, labels = data_iter.next()

for images, labels in train_loader:
    pass

# 5. Input pipeline for custom dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 1. Initialize file paths or a list of file names.
        pass
    def __getitem__(self, index):
        # 1. Read one data from file.
        # 2. Preprocess the data.
        # 3. Return a data pair.
        pass
    def __len__(self):
        # Change 0 to the total size of dataset.
        return 0
    
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset = custom_dataset, batch_size=64, shuffle=True)