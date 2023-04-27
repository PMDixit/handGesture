import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import opendatasets as od
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

def selectModel(model):
    global data_dir,target_num
    if(model=="Indian"):
        data_dir="..\\datasets\\data1\\train"
        target_num=36
    if(model=="American"):
        data_dir="..\\datasets\\segmentedImage"
        target_num=28
    TrainClass()
    
#-------------------------------------------------------------------
def TrainClass():
    train_tfms = tt.Compose([

                            tt.ToTensor(), 

                            ])

    valid_tfms = tt.Compose([tt.ToTensor()])

    global train_data
    train_data = ImageFolder(data_dir, transform=train_tfms)
    valid_data= ImageFolder(data_dir, transform=valid_tfms)
    test_data = ImageFolder(data_dir, transform=valid_tfms)

    #-------------------------------------------------------------------
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    valid_size = 0.15
    test_size = 0.10
    val_split = int(np.floor(valid_size * num_train))
    test_split = int(np.floor(test_size * num_train))
    valid_idx, test_idx, train_idx = indices[:val_split], indices[val_split:val_split+test_split], indices[val_split+test_split:]
    #-------------------------------------------------------------------
    batch_size = 250
    #------------------------------------------------------------------
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
        sampler=train_sampler, num_workers=2, pin_memory=True)
    valid_dl = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=2, pin_memory=True)
    #--------------------------------------------------------------------
    global device 
    device= get_default_device()
    #--------------------------------------------------------------------
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
#for computing check
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

#for dataloader
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

#---------------------------------------------------------------------
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
        
#----------------------------------------------------------------------------------------------------------
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

#model defenition
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64, pool=True) 
        self.conv2 = conv_block(64, 128, pool=True) 
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128)) 
        self.conv3 = conv_block(128, 256, pool=True) 
        self.conv4 = conv_block(256, 512, pool=True) 
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512)) 
        self.conv5 = conv_block(512, 512, pool=True) 
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.classifier(out)
        return out
#------------------------------------------------------------------------------------------------
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb).squeeze(0).softmax(0)
    # Pick index with highest probability
    preds  = yb.argmax().item()
    # Retrieve the class label
    return train_data.classes[preds]