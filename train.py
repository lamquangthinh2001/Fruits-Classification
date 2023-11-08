import torch
import torchvision
from torchvision import datasets, transforms

classes = ['cherry', 'strawberry', 'tomato']

import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.Resize([300,300]), # resize the image to its stock size of 300x300
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# read in and categorize data in given directory
trainset = datasets.ImageFolder('traindata', transform=transform) 
batch_size = 32

# create a trainloader with shuffle = True so as to make sure our training does not generalize to one class too much
trainloader = torch.utils.data.DataLoader(trainset, batch_size,
                                          shuffle=True)

# create our classification class 'Net'
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # take our 3 channel data and change to 6 channel with a kernal size of 5
        self.pool = nn.MaxPool2d(2, 2) # giving a pooling size of 2,2 means that our image size is halved
        self.conv2 = nn.Conv2d(6, 16, 5) # take our 6 channel data and change it to 16 channel data with same kernal size 5
        # based on kernal size, stride, padding and image size we get the below calculation
        self.fc1 = nn.Linear(16 * 72 * 72, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3) # must have 3 here as we want results for 3 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #ReLU activation function
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net() # create an instance of our classifier
import torch.optim as optim

criterion = nn.CrossEntropyLoss() # create the Cross Entropy loss function
optimizer = optim.Adamax(net.parameters(), lr=0.008, weight_decay=1e-4) # Adamax optimizer, L2 regularization using weight_decay

import numpy as np

# Create lists to store loss and accuracy for each epoch for both train and test sets

# Training and evaluation loop
for epoch in range(20):
    # Training
    net.train()
    running_loss_train = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Adding L1 regularization
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in net.parameters())
        loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()
        

print('Finished Training')

# save our model
PATH = 'model.pth'
torch.save(net.state_dict(), PATH)