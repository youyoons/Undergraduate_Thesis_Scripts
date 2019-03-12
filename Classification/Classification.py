import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import os
import math
try:
    import cPickle
except ImportError:
    import pickle as cPickle


#Set up Datastructure
class DatasetCSpine(Dataset):
    def __init__(self, directory_path, transform=None):
        self.directory_path = directory_path
        self.transform = transform
        
        self.image_dict = {}
        self.label_dict = {}
        
        path, dirs, files = next(os.walk(self.directory_path))
        
        self.files = files
        
        self.files.sort()

        for file in self.files:
            try:
                pre_segmented_input = cPickle.load(open(self.directory_path + "/" + file,"rb"),encoding = 'latin1')
            except:
                pre_segmented_input = cPickle.load(open(self.directory_path + "/" + file,"rb"))
        
            #print(file)
            pre_segmented_input = np.swapaxes(pre_segmented_input,0,2)
            #print(np.shape(pre_segmented_input))
            segmented_input = pre_segmented_input.astype(np.uint8).reshape((56,160,160))
            
            self.image_dict[file] = segmented_input
            
            ac_num = int(file.split("_")[2])
            
            #Label fractures with 1, no fractures with 0
            if ac_num < 100:
                self.label_dict[file] = 1
            else:
                self.label_dict[file] = 0
        
    def __len__(self):
        path, dirs, files = next(os.walk(self.directory_path))
        
        #return len(files)
        return len(files)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        
        #Read in pkl file depending on index
        file_index = self.files[index]
        image = self.image_dict[file_index]
        label = self.label_dict[file_index]
            
        return image, label


#Define ConvlNet Architecture
class CNN(nn.Module):
    def __init__(self, kernel, num_filters):
        super(CNN, self).__init__()
        padding = kernel // 2

        self.downconv1 = nn.Sequential(
            nn.Conv2d(56, num_filters, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),)
        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(2),)

        self.rfconv = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU())

        self.upconv1 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),)
        self.upconv2 = nn.Sequential(
            nn.Conv2d(num_filters, 3, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),)
            
        self.fc1 = nn.Sequential(
            nn.Linear(160*160*3,10),
            nn.ReLU(),)
            
        self.fc2 = nn.Sequential(
            nn.Linear(10,2),)

    def forward(self, x):
        #print(np.shape(x))
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(self.out3)
        self.out5 = self.upconv2(self.out4)
        #print(np.shape(self.out5))
        out = self.out5.reshape(self.out5.size(0),-1)
        #print(np.shape(out))
        self.out6 = self.fc1(out)
        #print(np.shape(self.out6))
        self.out_final = self.fc2(self.out6)
        #print(np.shape(self.out_final))
        #print("DONE")
        return self.out_final

#Define ConvlNet Architecture
class Simple_NN(nn.Module):
    def __init__(self, kernel, num_filters):
        super(Simple_NN, self).__init__()
        padding = kernel // 2

        self.downconv1 = nn.Sequential(
            nn.Conv2d(56, num_filters, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),)
        #Output is num_filters, 80, 80    
        
        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),)
        #Output is num_filters, 40, 40

            
        self.fc1 = nn.Sequential(
            nn.Linear(num_filters*40*40,10),
            nn.ReLU(),)
            
        self.fc2 = nn.Sequential(
            nn.Linear(10,2),)

    def forward(self, x):
        #print(np.shape(x))
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.out2.reshape(self.out2.size(0),-1)
        self.out4 = self.fc1(self.out3)
        self.out_final = self.fc2(self.out4)
        return self.out_final



#Start Main Function **********************************************************
os.chdir("C:\\Users\\yoons\\Documents\\ESC499\\Undergraduate_Thesis_Scripts")

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)


#There are 138 datasets in the training set

#print(np.shape(train_dataset.image_dict))
#print(train_dataset.__len__())
#print(np.shape(train_dataset.__getitem__(39)))


#*******************************************************************************
#*********************************** TRAINING **********************************
#*******************************************************************************
train_dataset = DatasetCSpine("Fracture_Classification_Data", transform = None)
train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True)

#Set architecture
kernel = 5
num_filters = 32

net = Simple_NN(kernel, num_filters).float()

#Set Loss Function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 12


#print(enumerate(train_loader,0))
#for i,data in enumerate(train_loader, 0):
#    print(i)


for epoch in range(num_epochs): 

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        print("i: ", i)
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #if i % 5 == 4:    # print every 10 mini-batches
        print('[%d, %3d] loss: %.3f' %(epoch + 1, i + 1, running_loss))
        running_loss = 0.0

print('Finished Training')


#*******************************************************************************
#*********************************** TESTING ***********************************
#*******************************************************************************
test_dataset = DatasetCSpine("Fracture_Classification_Data", transform = None)
test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

correct = 0
total = 0

print("\n\nStarting Testing")
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))



