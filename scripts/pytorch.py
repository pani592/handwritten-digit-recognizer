# This script contains code for training and testing of MNIST dataset from torchvision, using pytorch.
# Last updated: 18 April

from __future__ import print_function
from torch import nn, optim, cuda, Tensor
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import cv2 as cv
from skimage.color import rgb2gray

# Run on GPU if available
device = 'cuda' if cuda.is_available() else 'cpu'
# print(f'You are Using {device}')

# Training settings
input_size = 784 # 28x28 image size
num_classes = 10 # 0-9 digits
num_epochs = 10
batch_size = 64
learning_rate = 0.01

# MNIST Dataset - download.
train_dataset = datasets.MNIST(root='mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='mnist_data/', train=False, transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Display Training and Testing Images - NOTE: this needs to be shown on GUI.
def RandomNumberDisplay():
    '''When called, it displays a random sample from MNIST dataset with its label'''
    idx = random.randint(0, 60000)	
    x, label = train_dataset[idx] # x is a torch.Tensor (image) of size [1,28,28]
    plt.title('Example of a {}'.format(label))
    plt.axis('off')
    plt.imshow(x.numpy().squeeze(), cmap='gray') # can do gray_r for black on white

class ConvNet(nn.Module):
    ''' Covolutional Deep Learning Model - more accurate than inital MLP model with 5 layers '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5) # input channel dim 1 (grayscale), 10 output channels, 5 core size
        self.conv2 = nn.Conv2d(10, 20, 3) # 10 input channels, 20 output channels, 3 core size 
        self.fc1 = nn.Linear(20*10*10, 500) # 2000 input channels, 500 output channels
        self.fc2 = nn.Linear(500, 10) # 500 input channels, 10 output channels
    def forward(self,x):
        in_size = x.size(0) # in_size = batch size(n). x is as a tensor of n*1*28*28.
        x = F.relu(self.conv1(x)) # n*1*28*28 -> n*10*24*24 (28x28 image undergoes a convolution with a core of 5x5, and the output becomes 24x24)
        x = F.max_pool2d(x, 2, 2) # n*10*24*24 -> n*10*12*12 (2*2 pooling layer will be halved)
        x = F.relu(self.conv2(x)) # n*10*12*12 -> n*20*10*10 
        x = x.view(in_size, -1) # n*20*10*10 -> n*2000 
        x = F.relu(self.fc1(x)) # n*2000 -> n*500
        return F.log_softmax(self.fc2(x), dim=1) # n*500 -> n*10

model = ConvNet().to(device) # define an instance of the model
criterion = nn.CrossEntropyLoss() # loss criterion
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # optimizer

def train(epoch):
    '''Training the Model'''
    model.train() # set model to train mode
    for batch_idx, (images, labels) in enumerate(train_loader): # iterate through each batch
        images, labels = images.to(device), labels.to(device)   # send to cpu/gpu
        optimizer.zero_grad() # zero out optimizer gradients
        predictions = model(images) # pass batch of image tensors to model, return predictins for the batch
        loss = criterion(predictions, labels) # calculate loss by comparing prediction to actual label
        loss.backward() # backward pass
        optimizer.step() # use optimizer to modify model parameters
        if batch_idx % 100 == 0:  
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(epoch, batch_idx*len(images), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))

def test():
    '''Testing the Model'''
    model.eval() # set model to evaluate mode
    test_loss = 0
    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) # cpu or gpu
        predictions = model(images)
        test_loss += criterion(predictions, labels).item() # sum up batch loss
        pred = predictions.data.max(1, keepdim=True)[1]   # get the index of the max - gives prediction
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')
    return correct/len(test_loader.dataset)

def model_run():
    ''' runs model - training and testing for the number of epochs specified, and saves model'''
    since = time.time()
    for epoch in range(1,num_epochs):
        epoch_start = time.time()        
        train(epoch) # training 
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')      
        test() # Testing after each training round
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')   
    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
    # Save the model after completing training
    torch.save(model, './pytorch_model.pth')  # save model

def predict(tensor, model):
    ''' given a trained model and single image tensor, return prediction for the image'''
    model.eval() # set model to evaluate mode
    tensor = tensor.to(device)
    with torch.no_grad(): # recommended for speed
        prediction = model(tensor.float())
    probab = list(torch.exp(prediction).data.cpu().numpy()[0])
    pred = probab.index(max(probab))
    # pred = prediction.data.cpu().numpy().argmax() # another equivalent way
    return pred, probab

def view_classify(img, probab):
    ''' Function for viewing the image and it's prediction. Note: can display the real image if that's better visually'''
    # probab = probab.cpu().numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.cpu().resize_(1, 28, 28).numpy().squeeze(), cmap='gray')
    ax1.axis('off')
    ax2.barh(np.arange(10), probab)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

def recognize():
    ''' This will be moved to a better location later. Calls predict() from pytorch script'''
    model = torch.load('pytorch_model.pth') # load model
    img = cv.imread('digit_inv_28x28.jpg') # load image - [28,28,3]
    img = rgb2gray(img) # make grayscale - [28x28]
    tensor = torch.tensor(img)  # transform to tensor
    tensor = tensor.reshape(-1, 1, 28, 28) # make into form acceptable by model
    # tensor = tensor.flatten() # flatten
    pred, probab = predict(tensor, model) # use model to predict
    view_classify(tensor,probab) # plot image and probability
    return pred, probab

if __name__ == "__main__":
    ''' testing model - only runs when this script is run directly'''
    # model_run()  # trains for 10 epochs and displays gradual increase in accuracy, saves model.
    # model = torch.load('pytorch_model.pth') # load saved model
    # test()   
    recognize()