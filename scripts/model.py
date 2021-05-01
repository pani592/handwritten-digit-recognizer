# This file Contains the implementation of two models, including a Convolutional Neural Network, as well as helper functions for 
# training, testing, viewing MNIST, show probabilities, and making predictions. 
# Authors: Paulse Anithottam, Sidharth Varma
# imports
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
import pandas as pd

device = 'cuda' if cuda.is_available() else 'cpu' # Run on GPU if available

# Training settings
input_size = 784 # 28x28 image size
num_classes = 10 # 10 digits - classes
num_epochs = 20
batch_size = 64

# MNIST Dataset - download.
train_dataset = datasets.MNIST(root='mnist_data/', train=True, transform=transforms.ToTensor(), download=True) # 60000 images
test_dataset = datasets.MNIST(root='mnist_data/', train=False, transform=transforms.ToTensor()) # 10000 images

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Conv_Net_Model(nn.Module):
    ''' Convolutional Neural Network Model based on LeNet-5 network'''
    def __init__(self):
        super().__init__()
        self.convolutional1 = nn.Conv2d(1, 10, 5) # 1 input channel (grayscale), 10 output channels, core size 5
        self.convolutional2 = nn.Conv2d(10, 20, 3) # 10 input channels, 20 output channels, core size 3
        self.fullyconnected1 = nn.Linear(2000, 500) # 2000 input channels, 500 output channels
        self.fullyconnected2 = nn.Linear(500, num_classes) # 500 input channels, 10 output channels
    def forward(self,x):
        batch_size = x.size(0) # batch size(n). x is as a tensor of n*1*28*28.
        x = F.relu(self.convolutional1(x)) # n*1*28*28 -> n*10*24*24 (28x28 image undergoes convolution with a core of 5x5, and the output becomes 24x24)
        x = F.max_pool2d(x, 2, 2) # n*10*24*24 -> n*10*12*12 (2*2 pooling layer therefore halved)
        x = F.relu(self.convolutional2(x)) # n*10*12*12 -> n*20*10*10 
        x = x.view(batch_size, -1) # n*20*10*10 -> n*2000 
        x = F.relu(self.fullyconnected1(x)) # n*2000 -> n*500
        return F.log_softmax(self.fullyconnected2(x), dim=1) # n*500 -> n*10

class Neural_Net_Model(nn.Module):
    ''' Simple Feed Forward Neural Network Model provided in labs'''
    def __init__(self):
        super(Neural_Net_Model, self).__init__()
        self.l1 = nn.Linear(784, 520) # 784 input channels since the 28x28 image flattened has size 784
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10) # 10 outputs channels for 10 digits
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x)) 
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return F.log_softmax(self.l5(x), dim=1) # logsoftmax activation in final layer

model_1 = Conv_Net_Model().to(device) # define an instance of the model
model_2 = Neural_Net_Model().to(device)
criterion = nn.CrossEntropyLoss() # loss criterion
optimizer_1 = optim.SGD(model_1.parameters(), lr=0.01, momentum=0.9)  # optimizer
optimizer_2 = optim.SGD(model_2.parameters(), lr=0.01, momentum=0.9)  # optimizer

def train(epoch, input):
    '''Trains and saves the Model to files. Input = 1 means model 1 is chosen, input = 2 means model 2 is chosen.
    Inputs:
        epoch - current epoch number being run
        input - value corresponding to model choice
    Outputs:
        None (but saves model to file)
    '''  
    if input == 1:
        model = model_1
        optimizer = optimizer_1
    else: 
        model = model_2
        optimizer = optimizer_2
    model.train() # set model to train mode
    for batch_idx, (images, labels) in enumerate(train_loader): # iterate through each batch
        images, labels = images.to(device), labels.to(device)   # send to cpu/gpu
        optimizer.zero_grad() # zero out optimizer gradients
        predictions = model(images) # pass batch of image tensors to model, return predictins for the batch
        loss = criterion(predictions, labels) # calculate loss by comparing prediction to actual label
        loss.backward() # backward pass
        optimizer.step() # use optimizer to modify model parameters
        # if batch_idx % 100 == 0:  
        #     print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(epoch, batch_idx*len(images), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
    if input == 1:
        torch.save(model, './pytorch_model_1.pth')  # save model with name
    else: 
        torch.save(model, './pytorch_model_2.pth')  # save model with name

def test(input):
    '''Testing the Model to quantify accuracy and calculate confusion matrix.
    Inputs:
        input - value corresponding to model choice
    Outputs:
        model accuracy
        conusion matrix
    '''
    if input == 1:
        model = model_1
    else: 
        model = model_2
    model.eval() # set model to evaluate mode
    test_loss = 0
    correct = 0
    confusion_matrix = np.zeros((10, 10))
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) # cpu or gpu
        predictions = model(images)
        test_loss += criterion(predictions, labels).item() # sum up batch loss
        predictions_val = predictions.data.max(1, keepdim=True)[1]   # get the index of the max - gives prediction
        for t, p in zip(labels.view(-1), predictions_val.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1   # create confusion matrix to compare results
        correct += predictions_val.eq(labels.data.view_as(predictions_val)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    # calculate confusion matrix for analysis results, using pandas for legibility over normal np array
    df = pd.DataFrame(data=confusion_matrix)
    df.loc['Col_Total']= df.sum(numeric_only=True, axis=0)
    df.loc[:,'Row_Total'] = df.sum(numeric_only=True, axis=1)
    return correct/len(test_loader.dataset), df

def show_MNIST_examples():
    '''When called, this function plots and saves to file 35 random samples from the MNIST dataset with its label'''
    fig=plt.figure(figsize=(8,7))
    randnums = np.random.randint(0,60000,35) # array of length 35 with random numbers in required range
    for i in range(1, 36): # 35 images
        idx = randnums[i-1]
        x, label = train_dataset[idx] # x is a torch.Tensor (image) of size [1,28,28]
        fig.add_subplot(5, 7, i) # 2 rows 5 cols
        plt.title('{}'.format(label))
        plt.axis('off')
        plt.imshow(x.numpy().squeeze(), cmap='gray')
    plt.savefig('mnist_examples.jpg')  # save figure to folder
    plt.close(fig)

def predict(tensor, model):
    ''' Return the prediction and probability of classification of a certain image
    Inputs:
        tensor - pytorch tensor containing the digit to be predicted/ identified
        model - trained model used for evaluation
    Outputs:
        pred - prediction / digit output from model with highest probability
        probab - list of probabilities of each digit output from model 
    '''
    model.eval() # set model to evaluate mode
    tensor = tensor.to(device)
    with torch.no_grad(): # recommended for speed
        prediction = model(tensor.float())
    probab = list(torch.exp(prediction).data.cpu().numpy()[0]) # list of probabilities for each class
    pred = probab.index(max(probab)) # class with max probability is the predicted value
    # pred = prediction.data.cpu().numpy().argmax() # another equivalent way
    return pred, probab

def plot_probabilities(tensor, probab):
    ''' Function for plotting image of the handwritten digit and probabilities. Tensor - image input to model. Probab - array of probabilities for each class
    Inputs:
        tensor - tensor containing image data, which was input into the model for prediction
        probab - array of probabilities for each digit
    Returns: None ( but the figure is saved to file)
    '''
    fig, (ax0,ax1, ax2) = plt.subplots(figsize=(5,4), ncols=3)
    img = cv.imread('digit.jpg')
    ax0.imshow(img) # original image.
    ax0.set_title('Handwriting')
    ax0.axis('off')
    ax1.imshow(tensor.cpu().resize_(1, 28, 28).numpy().squeeze(), cmap='gray') # transform tensor to show on screen
    ax1.axis('off')
    ax1.set_title('Tensor Input to Model')
    ax2.barh(np.arange(10), probab) # plot probability array
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig('class_prob.jpg') # save plot to folder
    plt.close(fig)

def recognize(input):
    ''' This function imports the saved model + image, and calls predict() function, after some image manipulation including padding. 
    Inputs: 
        input - type int, either 1 or 0, corresponding to the model choice
    Outputs:
        pred - predicted digit returned from the model
        probab - array with probabilities for each digit returned from the model
    '''
    if input == 1:
        model = torch.load('pytorch_model_1.pth') # load model
    else: 
        model = torch.load('pytorch_model_2.pth') # load model

    img = cv.imread('digit_inv_20x20.jpg') # load image - [20x20x3]
    img = cv.copyMakeBorder(img.copy(), 4, 4, 4, 4, cv.BORDER_CONSTANT) # add padding to make 28x28
    img = rgb2gray(img) # make grayscale - [28x28]
    # ret,img = cv.threshold(img,0.8,1,cv.THRESH_BINARY) # a threshold filter may increase accuracy of prediction, but inconsistent hence commented out.
    tensor = torch.tensor(img)  # transform to tensor [1x28x28]
    tensor = tensor.reshape(-1, 1, 28, 28) # make into form acceptable by model [1x1x28x28]
    pred, probab = predict(tensor, model) # use model to predict, return prediction and probability array
    plot_probabilities(tensor,probab) # plot images and probability and save to file
    return pred, probab

if __name__ == "__main__":
    ## if this script is run directly, it trains both models and returns the confusion matrix for each
    for epoch in range(1,21):  # 20 epochs
        train(epoch = epoch, input = 1)
    acc1, conf_matrix1 = test(1)
    print(acc1)
    print(conf_matrix1)

    # model 2
    for epoch in range(1,21):  # 20 epochs
        train(epoch = epoch, input = 2)
    acc2, conf_matrix2 = test(2)
    print(acc2)
    print(conf_matrix2)