# This script contains code for training and testing of MNIST dataset from torchvision, using pytorch.
# Last updated: 15 April

from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import torch

# Run on GPU if available
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST Model on {device}\n{"=" * 44}')

# Training settings
input_size = 784 # 28x28 image size
num_classes = 10 # 0-9 digits
num_epochs = 10
batch_size = 64
learning_rate = 0.01

# MNIST Dataset
train_dataset = datasets.MNIST(root='mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='mnist_data/', train=False, transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# for showing training and test data sets
examples = iter(train_loader)
samples,labels = examples.next()
print(samples.shape, labels.shape)  # samples: torch.Size([64, 1, 28, 28]) labels: torch.Size([64]) - Note: 64 because of batch size = 64.

# num_of_images = 60
# for index in range(num_of_images):
#     plt.subplot(6, 10, index+1)
#     plt.axis('off')
#     plt.imshow(samples[index].numpy().squeeze(), cmap='gray_r')  #gray r has white bg and black font
# plt.show()

# Multilayer Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, 520)    
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, num_classes)
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x)) 
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

# Covolutional Model (more accurate)
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5) #  1 input channels, 10 output channels, 5 core size
        self.conv2 = nn.Conv2d(10, 20, 3) # 10 input channels, 20 output channels, 3 core size 
        self.fc1 = nn.Linear(20*10*10, 500) # 2000 input channels, 500 output channels
        self.fc2 = nn.Linear(500, 10) # 500 input channels, 10 output channels
    def forward(self,x):
        in_size = x.size(0) # in_size= value of BATCH_SIZE. The input x can be regarded as a tensor of n*1*28*28.
        x = F.relu(self.conv1(x)) # batch*1*28*28 -> batch*10*24*24 (28x28 image undergoes a convolution with a core of 5x5, and the output becomes 24x24)
        x = F.max_pool2d(x, 2, 2) # batch*10*24*24 -> batch*10*12*12 (2*2 pooling layer will be halved)
        x = F.relu(self.conv2(x)) # batch*10*12*12 -> batch*20*10*10 
        x = x.view(in_size, -1) # batch*20*10*10 -> batch*2000 
        x = F.relu(self.fc1(x)) # batch*2000 -> batch*500
        return F.log_softmax(self.fc2(x), dim=1) # batch*500 -> batch*10

# intialise model, define loss and optimiser
# model = NeuralNet(input_size, hidden_size, num_classes).to(device)
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # mom originally 0.5. Could use Adam but SGD is better.

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)   # cpu or gpu
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:  # print every 100 steps - reduce steps to create progress bars later on.
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device) # cpu or gpu
        output = model(data)
        test_loss += criterion(output, target).item()         # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]         # get the index of the max
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')


if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, 2):
        epoch_start = time.time()
        train(epoch) # TRAIN
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test() # TEST
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')

    #Save the model after training
    # torch.save(model, './my_model.pth')