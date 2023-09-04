# Clear_quote_exercise1
data preprocessing , model , tflite conversion

Model architecture –
CONV1 -> BN1 -> RELU1 -> POOL1 -> CONV2 -> RELU2 -> CONV3 -> RELU3 -> FC1
CONV= convolutional layer
BN = batch normalisation
POOL= max pooling layer
FC= fully connected layer


Dataloader – 
 The data loader takes the directory path which contains all the image folders
The dataframe format containing image names and folder names for loading data 
Transform function


Imports required to run the model –


import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.autograd import Variable




To run the model  create a dataframe using the following code-
data_dir=' Path to the directory containing the folders with images '
folders=os.listdir(data_dir)
ds = pd.DataFrame(columns=["directory_name","image_name",])
for k in folders:
    df = pd.read_json(f"D:\\exercise1\\exercise_1\\{k}\\via_region_data.json")
    images = df.columns
    size = len(images)
    
    ds.loc[len(ds.index)] = [k,images[i]]





Creating dataloader – 
class CustomDataset(Dataset):
    def __init__(self, root_dir, dataframe, transform=None):
        self.root_dir = root_dir
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0], self.dataframe.iloc[idx, 1])
        image = Image.open(img_name)
        
        
        if self.transform:
            image = self.transform(image)
        
        return image, label




Define the transform for the image dataset -

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to a fixed size
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])


dataset = CustomDataset(root_dir=data_dir, dataframe=ds, transform=transform)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')




Defining the model –
class convnet(nn.Module):
    def __init__(self,num_classes=6):
        super(convnet,self).__init__()
        
        #(32,3,224,224)
        self.conv1=nn.Conv2d(3,12,kernel_size=3,stride=1,padding=1)
        #(32,12,224,224)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #(32,12,224,224)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)
        #(32,12,112,112)
        
        self.conv2=nn.Conv2d(12,20,kernel_size=3,stride=1,padding=1)
        #(32,20,112,112)
        self.relu2=nn.ReLU()
        
        
        self.conv3=nn.Conv2d(20,32,kernel_size=3,stride=1,padding=1)
        #(32,32,112,112)
        self.bn3=nn.BatchNorm2d(num_features=32)
        
        self.relu3=nn.ReLU()
        #(32,32,112,112)
        
        self.fc1=nn.Linear(32*112*112,6)
        #self.relu4=nn.ReLU()
        #self.fc2=nn.Linear(64,6)
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        
        
        x=self.conv2(x)
        x=self.relu2(x)
        
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu3(x)
        
        x=x.view(-1,32*112*112)
        
        x=self.fc1(x)
        #x=self.relu4(x)
        #x=self.fc2(x)
        
        
        return x



creating model instance –

model=convnet(num_classes=6).to(device)

loading the model params –
file="model.pth"
torch.save(model.state_dict(),file)
model_loaded= convnet(num_classes=6)
model_loaded.load_state_dict(torch.load(file))



also download and save the model.pth file in the working directory where the model is being created to directly import the model weights and params


then the dataloader can be used to input images to the model to get the vector form of output s
Where the one hot vectors for each class are as follows 
Front =[1,0,0,0,0,0]
Rear=[0,1,0,0,0,0]
Front right=[0,0,1,0,0,0]
Rear right=[0,0,0,1,0,0]
Rear left=[0,0,0,0,1,0]
Front left=[0,0,0,0,0,1]

Label encoded value can be obtained using –
Label_predicted = np.argmax(np.array(output_vector))
 
Where  the label encodings for different classes are as follows –
Front =0
Rear =1
Front right =2
Rear right =3
Rear left =4
Front left =5



Tflite model conversion and usage –
To convert the model from pytorch to tflite use the code as in the jupyter notebook 
Make sure to download and save the model.pth , onnx_model.onnx , tf_model.keras, tflite_model.tflite files in the working directory before using

Use the inference code to further load and use the model in any of the above format –
Onnx 
Tf 
Tflite



