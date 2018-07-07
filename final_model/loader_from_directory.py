# to measure run-time
import time

# to shuffle data
import random

# for plotting
import matplotlib.pyplot as plt

# pytorch stuff
import torch
import torch.nn as nn
import torch.optim as optim

# for csv dataset
import torchvision
import csv
import os
import pandas as pd 
from urllib import request

# import statements for iterating over csv file
import cv2
import numpy as np
import urllib.request

# to get the alphabet
import string
from PIL import Image

# generate the targets 
# the targets are one hot encoding vectors

alphabet = list(string.ascii_lowercase)
target = {}

# Initalize a target dict that has letters as its keys and empty one-hot encoding vectors of size 37 as its values
for letter in alphabet: 
    target[letter] = [0] * 37

# Do the one-hot encoding for each letter now 
curr_pos = 0 
for curr_letter in target.keys():
    target[curr_letter][curr_pos] = 1
    curr_pos += 1  

# extra symbols 
symbols = ["space", "number", "period", "comma", "colon", "apostrophe", "hyphen", "semicolon", "question", "exclamation", "capitalize"]

# create vectors
for curr_symbol in symbols:
    target[curr_symbol] = [0] * 37

# create one-hot encoding vectors
for curr_symbol in symbols:
    target[curr_symbol][curr_pos] = 1
    curr_pos += 1

# collect all data from the csv file
data = []

for tgt in os.listdir("dataset"):
    if not tgt == ".DS_Store":
        for folder in os.listdir("dataset/" + tgt + "/Uploaded"):
                if not folder == ".DS_Store":
                    for filename in os.listdir("dataset/" + tgt + "/Uploaded/" + folder):
                        if not filename == ".DS_Store":
                        # store the image and label
                            picture = []
                            curr_target = target[tgt]
                            image = Image.open("dataset/" + tgt + "/Uploaded/" + folder + "/" + filename)
                            image = image.convert('RGB')
                            # f.show()
                            image = np.array(image)
                            # resize image to 28x28x3
                            image = cv2.resize(image, (28, 28))
                            # normalize to 0-1
                            image = image.astype(np.float32)/255.0
                            image = torch.from_numpy(image)
                            picture.append(image)
                            # convert the target to a long tensor
                            curr_target = torch.Tensor([curr_target])
                            picture.append(curr_target)
                            # append the current image & target
                            data.append(picture)

# create a dictionary of all the characters 
characters = alphabet + symbols

index2char = {}
number = 0
for char in characters: 
    index2char[number] = char
    number += 1

# find the number of each character in a dataset
def num_chars(dataset, index2char):
    chars = {}
    for _, label in dataset:
        char = index2char[int(torch.argmax(label))]
        # update
        if char in chars:
            chars[char] += 1
        # initialize
        else:
            chars[char] = 1
    return chars

# Create dataloader objects

# shuffle all the data
random.shuffle(data)

# batch sizes for train, test, and validation
batch_size_train = 30
batch_size_test = 30
batch_size_validation = 30

# splitting data to get training, test, and validation sets
# change once get more data
# 1600 for train
train_dataset = data[:22000]
# test has 212
test_dataset = data[22000:24400]
# validation has 212
validation_dataset = data[24400:]

# create the dataloader objects
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size_validation, shuffle=True)

# to check if a dataset is missing a char
test_chars = num_chars(test_dataset, index2char)

num = 0
for char in characters:
    if char in test_chars:
        num += 1
    else:
        break
print(num) 


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            #3x28x28
            nn.Conv2d(in_channels=3, 
                      out_channels=16, 
                      kernel_size=5,
                      stride=1, 
                      padding=2),
            # batch normalization
            # nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True), 
            #16x28x28
            nn.MaxPool2d(kernel_size=2),
            #16x14x14
            nn.LeakyReLU()
        )
        #16x14x14
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, 
                      out_channels=32, 
                      kernel_size=5,
                      stride=1, 
                      padding=2),
            # batch normalization
            # nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True), 
            #32x14x14
            nn.MaxPool2d(kernel_size=2),
            #32x7x7
            nn.LeakyReLU()
        ) 
        # linearly 
        self.block3 = nn.Sequential(
            nn.Linear(32*7*7, 100),
            # batch normalization
            # nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 37)
        )
        #1x37
    
    def forward(self, x): 
        out = self.block1(x)
        out = self.block2(out)
        # flatten the dataset
        out = out.view(-1, 32*7*7)
        out = self.block3(out)
        
        return out


# convolutional neural network model
model = CNN()

# print summary of the neural network model to check if everything is fine. 
print(model)
print("# parameter: ", sum([param.nelement() for param in model.parameters()]))

# setting the learning rate
learning_rate = 1e-4

# Using a variable to store the cross entropy method
criterion = nn.CrossEntropyLoss()

# Using a variable to store the optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# list of all train_losses 
train_losses = []

# list of all validation losses 
validation_losses = []

# for loop that iterates over all the epochs
num_epochs = 10
for epoch in range(num_epochs):
    
    # variables to store/keep track of the loss and number of iterations
    train_loss = 0
    num_iter_train = 0

    # train the model
    model.train()
    
    # Iterate over train_loader
    for i, (images, labels) in enumerate(train_loader):  
        # need to permute so that the images are of size 3x28x28 
        # essential to be able to feed images into the model
        images = images.permute(0, 3, 1, 2)

        # Zero the gradient buffer
        # resets the gradient after each epoch so that the gradients don't add up
        optimizer.zero_grad()  

        # Forward, get output
        outputs = model(images)
        
        # convert the labels from one hot encoding vectors into integer values 
        labels = labels.view(-1, 37)
        y_true = torch.argmax(labels, 1)

        # calculate training loss
        loss = criterion(outputs, y_true)
        
        # Backward (computes all the gradients)
        loss.backward()
        
        # Optimize
        # loops through all parameters and updates weights by using the gradients 
        # takes steps backwards to optimize (to reach the minimum weight)
        optimizer.step()
        # update the training loss and number of iterations
        train_loss += loss.data[0]
        num_iter_train += 1

    print('Epoch: {}'.format(epoch+1))
    print('Training Loss: {:.4f}'.format(train_loss/num_iter_train))
    # append training loss over all the epochs
    train_losses.append(train_loss/num_iter_train)

    # evaluate the model
    model.eval()
    
    # variables to store/keep track of the loss and number of iterations
    validation_loss = 0
    num_iter_validation = 0
    
    # Iterate over validation_loader
    for i, (images, labels) in enumerate(validation_loader):  
        # need to permute so that the images are of size 3x28x28 
        # essential to be able to feed images into the model
        images = images.permute(0, 3, 1, 2)

        # Forward, get output
        outputs = model(images)

        # convert the labels from one hot encoding vectors to integer values
        labels = labels.view(-1, 37)
        y_true = torch.argmax(labels, 1)
        
        # calculate the validation loss
        loss = criterion(outputs, y_true)

        # update the training loss and number of iterations
        validation_loss += loss.data[0]
        num_iter_validation += 1

    print('Validation Loss: {:.4f}'.format(validation_loss/num_iter_validation))
    # append all validation_losses over all the epochs
    validation_losses.append(validation_loss/num_iter_validation)

    num_iter_test = 0
    correct = 0
    
    # Iterate over test_loader
    for images, labels in test_loader:  

        # need to permute so that the images are of size 3x28x28 
        # essential to be able to feed images into the model
        images = images.permute(0, 3, 1, 2)

        # Forward
        outputs = model(images)

        # convert the labels from one hot encoding vectors into integer values 
        labels = labels.view(-1, 37)
        y_true = torch.argmax(labels, 1)

        # find the index of the prediction
        y_pred = torch.argmax(outputs, 1).type('torch.FloatTensor')
        
        # convert to FloatTensor
        y_true = y_true.type('torch.FloatTensor')

        # find the mean difference of the comparisons
        correct += torch.sum(torch.eq(y_true, y_pred).type('torch.FloatTensor'))

    print('Accuracy on the test set: {:.4f}%'.format(correct/len(test_dataset) * 100))
    print()


# learning curve function
def plot_learning_curve(train_losses, validation_losses):
    # plot the training and validation losses
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.plot(train_losses, label="training")
    plt.plot(validation_losses, label="validation")
    plt.legend(loc=1)


# plot the learning curve
plt.title("Learning Curve (Loss vs Number of Epochs)")
plot_learning_curve(train_losses, validation_losses)

torch.save(model.state_dict(), "model.pth")

