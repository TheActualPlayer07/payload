import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam, AdamW
from sklearn.metrics import f1_score

# Improved Data Augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight translation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define Directories
train_dir = '/home/goodarth/Desktop/IEEE SPECTRO/Soil types'
test_dir = '/home/goodarth/Desktop/IEEE SPECTRO/Soil_types_test'

# Load Dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Here defined a CNN architecture by creating a class using nn.module from pytorch
# nn.module serves as a base class for all neuralnets, providing an extensive franework for defining and organizing the layers and functions of a neural network
class NeuralNetCNN(nn.Module):
    
    # this is the constructor method/function whichgets called on making every new instance of the class.
    def __init__(self, num_class_images = 6):
        
        # Initializes the paraent class
        # This is a necessary step alloing pytorch to initialise settings and proerties in the nn.module
        super(NeuralNetCNN,self).__init__()

        #Output size after convolution filter
        #((w-f+2p)/s) + 1
        # w = width image, f = filter width , p = padding width, s = stride
        # conv1/2/3 is notation for the nth CNN
        # in_channels specifies the number of colour channels used(like RGB)
        # out_channels specifies the number of filters used/ the number of resulting output channels
        # kernel_size specifies filter dimension size
        # stride specifies the stride of the filter(the number of steps it takes while moving the filter across the image)
        # padding specifies the padding size of the filter around the image, which is necessary to prevent the image from shrinking, and lead to quality drop in further steps
        # bn1/2/3 represent batch normalisation function which normalises the data in every layer post output
        # relu1/2/3 represent the activation function used in every layer post output(ReLU)


        # FIRST layer of the CNN
        # Input shape = (256,3,128,128)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        # Input shape = (256,32,128,128)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        # Input shape = (256,32,128,128)
        self.relu1 = nn.ReLU()
        # Input shape = (256,32,128,128)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Now shape = (256,32,64,64)

        # SECOND layer of CNN
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        # Now shape = (256,64,64,64)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        # Now shape = (256,64,64,64)
        self.relu2 = nn.ReLU()
        # Now shape = (256,64,64,64)
       
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Now shape = (256,64,32,32)

        # THIRD layer of CNN
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        # Now shape = (256,128,32,32)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        # Now shape = (256,128,32,32)
        self.relu3 = nn.ReLU()
        # Now shape = (256,128,32,32)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Now shape = (256,128,16,16)


        # # FOURTH layer of CNN
        # self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        # # Now shape = (256,256,4,4)
        # self.bn3 = nn.BatchNorm2d(num_features=32)
        # # Now shape = (256,256,4,4)
        # self.relu3 = nn.ReLU()
        # # Now shape = (256,256,4,4)

        # self.pool = nn.MaxPool2d(kernel_size=2)
        # # Now shape = (256,256,2,2)

        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)  # Regularization

        # this is the fully connected layer, which makes a single column matrix by stacking all data of the final matrix post the 3rd layer
        self.fc2 = nn.Linear(512,out_features=num_class_images)

    # This is the forward pass of the network, it is the function that is called when the network is run
    def forward(self,input):
        # applying all layers in order
        output =self.conv1(input)
        output =self.bn1(output)
        output =self.relu1(output)
        output =self.pool(output)
        output =self.conv2(output)
        output =self.bn2(output)
        output =self.relu2(output)
        output =self.pool(output)
        output =self.conv3(output)
        output =self.bn3(output)
        output =self.relu3(output)
        output =self.pool(output)

        output = output.view(-1,128*16*16)

        output =self.fc1(output)
        output =self.relu1(output)
        output =self.dropout(output)
        output =self.fc2(output)
        

        return output

# This line sets the computation device to GPU if available, otherwise CPU.
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creates an instant of the defined CNN achitecture and transfers it to chosen device
model = NeuralNetCNN(num_class_images = 6).to(device)

# made use of adam optimizer
# model.parameters() passes the model’s learnable parameters (weights and biases) to the optimizer.
# learning rate is the step size for adjusting weights with each optimization step. In Adam, it controls the general rate of updates to the model's parameters.
# weight decay is a form of regularisation which helps prevent overfitting, Setting weight_decay=0.0001 means that a small penalty is applied to the model weights, encouraging smaller weights and potentially improving the model’s generalization to new data.
optimizer = AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)

# loss function used is crossentropyloss
loss_function = nn.CrossEntropyLoss()


num_epochs = 30


#train_count = len(glob.glob(train_dir+'/**/*.jpg',recursive=True))
#test_count = len(glob.glob(test_dir+'/**/*.jpg'))
#print(test_count,train_count)
#print(test_dir,train_dir)




best_accuracy = 0.0
train_losses_per_epoch = []

#  for loop which runs for a specified number of epochs.
for epoch in range(num_epochs):
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    train_predictions = []
    train_labels = []

    # for loop over batches in training data
    for batch_index, (images, labels) in enumerate(train_loader):
        # helps clear previous grads before backprop
        optimizer.zero_grad()


        # print(f"Batch Index: {batch_index}")
        
        # passes images through the model
        outputs = model(images.to(device))
        # loss calculation between predictions and true 
        loss = loss_function(outputs,labels.to(device))
        # gradient computation (backprop)
        loss.backward()
        # updation of parameters
        optimizer.step()



        # calculating accuracy
        train_loss = train_loss + loss.cpu().data*images.size(0)
        _,prediction = torch.max(outputs.data,1)
        train_accuracy+=int(torch.sum(prediction==labels.data.to(device)))


        # Store predictions and labels for F1 score
        train_predictions.extend(prediction.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_f1 = f1_score(train_labels, train_predictions, average='weighted')
    
    # calculating accuracy over entire dataset
    train_accuracy = train_accuracy/len(train_loader.dataset)
    train_loss = train_loss/len(train_loader.dataset)

    train_losses_per_epoch.append(train_loss)


    model.eval()
    test_accuracy = 0.0
    test_predictions = []
    test_labels = []

    
    for batch_index, (images, labels) in enumerate(test_loader):
        outputs = model(images.to(device))
        # torch.max(outputs,data,1) gets the predicted class from the data.
        _,prediction = torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data.to(device)))

        # Store predictions and labels for F1 score
        test_predictions.extend(prediction.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        

    test_f1 = f1_score(test_labels, test_predictions, average='weighted')
        
    test_accuracy = test_accuracy/len(test_loader.dataset)

   
    print(f'Epoch: {epoch + 1}, '
          f'Train Loss: {train_loss:.4f}, '
          f'Train Accuracy: {train_accuracy:.4f}, '
          f'Train F1 Score: {train_f1:.4f}, '
          f'Test Accuracy: {test_accuracy:.4f}, '
          f'Test F1 Score: {test_f1:.4f}')
    
    #saving the best model if accuracy improves over epochs.
    if test_accuracy>best_accuracy:
        torch.save(model.state_dict(),'best_epoch.model')
        best_accuracy = test_accuracy

        
