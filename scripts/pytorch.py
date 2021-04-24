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
num_classes = 10 # 10 digits - classes
num_epochs = 10
batch_size = 64
learning_rate = 0.01

# MNIST Dataset - download.
train_dataset = datasets.MNIST(root='mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='mnist_data/', train=False, transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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
        # if batch_idx % 100 == 0:  
        #     print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(epoch, batch_idx*len(images), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))

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
    # print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
    #       f'({100. * correct / len(test_loader.dataset):.0f}%)')
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

def showMNISTExamples():
    '''When called, it plots and saves 10 random samples from MNIST dataset with its label'''
    fig=plt.figure()
    for i in range(1, 11): # 10 images
        idx = random.randint(0, 60000)	
        x, label = train_dataset[idx] # x is a torch.Tensor (image) of size [1,28,28]
        fig.add_subplot(2, 5, i) # 2 rows 5 cols
        plt.title('{}'.format(label))
        plt.axis('off')
        plt.imshow(x.numpy().squeeze(), cmap='gray')
    plt.savefig('mnist_examples.jpg')
    plt.close(fig)

def plot_probabilities(tensor, probab):
    ''' Function for plotting and saving the handwritten digit and graph of probabilities.'''
    fig, (ax0,ax1, ax2) = plt.subplots(figsize=(5,4), ncols=3)
    img = cv.imread('digit.jpg')
    ax0.imshow(img) # original image.
    ax0.set_title('Input')
    ax0.axis('off')
    ax1.imshow(tensor.cpu().resize_(1, 28, 28).numpy().squeeze(), cmap='gray') # transfor tensor to show on screen
    ax1.axis('off')
    ax1.set_title('Tensor')
    ax2.barh(np.arange(10), probab) # plot probability array
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig('class_prob.jpg')
    plt.close(fig)

def recognize():
    ''' This function imports the saved model + image, and makes prediction.'''
    model = torch.load('pytorch_model.pth') # load model
    img = cv.imread('digit_inv_20x20.jpg') # load image - [20x20x3]
    img = cv.copyMakeBorder(img.copy(), 4, 4, 4, 4, cv.BORDER_CONSTANT) # add padding to make 28x28
    img = rgb2gray(img) # make grayscale - [28x28]
    # ret,img = cv.threshold(img,0.5,1,cv.THRESH_BINARY) # threshold increases accuracy of prediction
    tensor = torch.tensor(img)  # transform to tensor [1x28x28]
    tensor = tensor.reshape(-1, 1, 28, 28) # make into form acceptable by model [1x1x28x28]
    pred, probab = predict(tensor, model) # use model to predict, return prediction and probability array
    plot_probabilities(tensor,probab) # plot images and probability and save to file
    return pred, probab

if __name__ == "__main__":
    ''' testing model - only runs when this script is run directly'''
    # model_run()  # trains for 10 epochs and displays gradual increase in accuracy, saves model.
    # recognize()
    showMNISTExamples()
