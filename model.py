import os
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt

def selectModel(model):
    global data_dir,target_num
    #selecting the dataset
    if(model=="Indian"):
        data_dir=os.path.join("..","datasets","data1","train")
        target_num=36
    if(model=="American"):
        data_dir=os.path.join("..","datasets","segmentedImage")
        target_num=28
    TrainClass()

def TrainClass():
    #creationg the image label pair
    train_tfms = tt.Compose([tt.ToTensor()])
    global train_data
    train_data = ImageFolder(data_dir, transform=train_tfms)
    global device 
    device= get_default_device()

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

#function for predicting the label
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb).squeeze(0).softmax(0)
    # Pick index with highest probability
    preds  = yb.argmax().item()
    # Retrieve the class label
    return train_data.classes[preds]