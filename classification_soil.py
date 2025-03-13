import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import pathlib
from torch.utils.data import DataLoader
from torch.optim import Adam
import glob


# To make a sequence of image transformations before processing
# This includes converting image to tensors, normalising pixel values and resizing images
transform = transforms.Compose([
    # Convert image to a PyTorch tensor:(basically scaling it down to 0 to 1)
    transforms.ToTensor(),
    # This function standardizes pixel values of imgs using given mean and std deviation as parameters(normalized_pixel = pixel-mean/std)(first parameter is for mean, second for std deviaiton)
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ,
    # This function resizes the image to the specified size
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15)
])



# Here I defined directories for train and test data
train_dir = '/home/goodarth/Desktop/IEEE SPECTRO/Soil types'
test_dir = '/home/goodarth/Desktop/IEEE SPECTRO/Soil_types_test'

# Loaded training and test datasets using ImageFolder, pointing to a specific local folder.
# THe images are also transformed as mentioned above as they are loaded
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)


# Create data loaders for training and test data
# Data Loaders enable optimised loading of datasets for training and testing(by batching,shufflinf and parallel loading)
# The batch size is set to 5, which means at once 5 images are loaded
# Shuffle parameter mentions if the daataset is shuffled or not before each epoch
# num_workers parameter controls the number of CPU processes that load data in parallel while the model trains
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True, num_workers=2)

# Here i defined the categories of images in the dataset for classification purposes
class_images =['Black Soil','Cinder Soil','Laterite Soil','Peat Soil','Yellow Soil']


# Here defined a CNN architecture by creating a class using nn.module from pytorch
# nn.module serves as a base class for all neuralnets, providing an extensive franework for defining and organizing the layers and functions of a neural network
class NeuralNetCNN(nn.Module):
    # this is the constructor method/function whichgets called on making every new instance of the class.
    def __init__(self, num_class_images = 5):
        # Initializes the paraent class
        # This is a necessary step alloing pytorch to initialise settings and proerties in the nn.module
        super(NeuralNetCNN,self).__init__()

        #Output size after convolution filter
        #((w-f+2p)/s) + 1
        # w = width image, f = filter width , p = padding width, s = stride
        # conv1/2/3 is notation for the nth CNN
        # in_channels specifies the number of colour channels used( like RGB)
        # out_channels specifies the number of filters used/ the number of resulting output channels
        # kernel_size specifies filter dimension size
        # stride specifies the stride of the filter(the number of steps it takes while moving the filter across the image)
        # padding specifies the padding size of the filter around the image, which is necessary to prevent the image from shrinking, and lead to quality drop in further steps
        # bn1/2/3 represent batch normalisation function which normalises the data in every layer post output
        # relu1/2/3 represent the activation function used in every layer post output(ReLU)


        # FIRST layer of the CNN
        # Input shape = (256,3,150,150)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        # Now shape = (256,12,150,150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Now shape = (256,12,150,150)
        self.relu1 = nn.ReLU()
        # Now shape = (256,12,150,150)

        self.dropout = nn.Dropout(p=0.5)

        
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Now shape = (256,12,75,75)

        # SECOND layer of CNN
        self.conv2 = nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        # Now shape = (256,20,75,75)
        self.bn2 = nn.BatchNorm2d(num_features=20)
        # Now shape = (256,20,75,75)
        self.relu2 = nn.ReLU()
        # Now shape = (256,20,75,75)


        # THIRD layer of CNN
        self.conv3 = nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        # Now shape = (256,32,75,75)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Now shape = (256,32,75,75)
        self.relu3 = nn.ReLU()
        # Now shape = (256,32,75,75)

        # this is the fully connected layer, which makes a single column matrix by stacking all data of the final matrix post the 3rd layer
        self.fc = nn.Linear(in_features=32*75*75,out_features=num_class_images)

    # This is the forward pass of the network, it is the function that is called when the network is run
    def forward(self,input):
        # applying all layers in order
        output =self.conv1(input)
        output =self.bn1(output)
        output =self.relu1(output)

        output = self.dropout(output)
        
        output =self.pool(output)
        output =self.conv2(output)
        output =self.bn2(output)
        output =self.relu2(output)
        output =self.conv3(output)
        output =self.bn3(output)
        output =self.relu3(output)

        output = output.view(-1,32*75*75)
        output =self.fc(output)

        return output
    
    # This line sets the computation device to GPU if available, otherwise CPU.
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creates an instant of the defined CNN achitecture and transfers it to chosen device
model = NeuralNetCNN(num_class_images = 5).to(device)

# made use of adam optimizer
# model.parameters() passes the model’s learnable parameters (weights and biases) to the optimizer.
# learning rate is the step size for adjusting weights with each optimization step. In Adam, it controls the general rate of updates to the model's parameters.
# weight decay is a form of regularisation which helps prevent overfitting, Setting weight_decay=0.0001 means that a small penalty is applied to the model weights, encouraging smaller weights and potentially improving the model’s generalization to new data.
optimizer = Adam(model.parameters(),lr=0.001,weight_decay=0.0001)

# loss function used is crossentropyloss
loss_function = nn.CrossEntropyLoss()


num_epochs = 50



#train_count = len(glob.glob(train_dir+'/**/*.jpg',recursive=True))
#test_count = len(glob.glob(test_dir+'/**/*.jpg'))
#print(test_count,train_count)
#print(test_dir,train_dir)


best_accuracy = 0.0
#  for loop which runs for a specified number of epochs.
for epoch in range(num_epochs):
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    # for loop over batches in training data
    for batch_index, (images, labels) in enumerate(train_loader):
        # helps clear previous grads before backprop
        optimizer.zero_grad()
        # passes images through the model
        outputs = model(images.to(device))
        # loss calculation between predictions and true labels
        loss = loss_function(outputs,labels)
        # gradient computation (backprop)
        loss.backward()
        # updation of parameters
        optimizer.step()



        # calculating accuracy
        train_loss = train_loss + loss.cpu().data*images.size(0)
        _,prediction = torch.max(outputs.data,1)
        train_accuracy+=int(torch.sum(prediction==labels.data))
    # calculating accuracy over entire dataset
    train_accuracy = train_accuracy/len(train_loader.dataset)
    train_loss = train_loss/len(train_loader.dataset)


    model.eval()
    test_accuracy = 0.0
    for batch_index, (images, labels) in enumerate(test_loader):
        outputs = model(images.to(device))
        # torch.max(outputs,data,1) gets the predicted class from the data.
        _,prediction = torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))
    test_accuracy = test_accuracy/len(test_loader.dataset)

    print(f'Epoch: {epoch + 1} Train Loss: {train_loss:.4f} Train Accuracy: {train_accuracy:.4f} Test Accuracy: {test_accuracy:.4f}')
    
    # print('Epoch: '+str(epoch)+'Train Loss: '+str(float(train_loss))+'Train Accuracy: '+str(float(train_accuracy))+'Test Accuracy: '+str(float(test_accuracy)))

    #saving the best model if accuracy improves over epochs.
    if test_accuracy>best_accuracy:
        torch.save(model.state_dict(),'best_epoch.model')
        best_accuracy = test_accuracy