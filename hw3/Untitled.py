#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


# # lettuce begin; part 1

# In[10]:


# load model and datset
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
criterion = nn.CrossEntropyLoss()
# normalize data; mean 0
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,0,0), (0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# shuffle training data points
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[7]:


dataiter = iter(trainloader)
images, labels = dataiter.next()
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

print(images[0].size())


# In[3]:


# get subset; randomly sample 1k/5k from each class
train_subset = {k: [] for k in np.arange(10)} # indices of images
train_cntr = {k: 0 for k in np.arange(10)} # counts how many samples of a given class we have
agg_cnt = 0
for i in range(len(trainset)):
    dp_label = trainset[i][1]
    if train_cntr[dp_label] < 1000: # if there aren't 1000 samples for this class already
        train_subset[dp_label].append(i) # add it
        train_cntr[dp_label] += 1 # and increment counter
        agg_cnt += 1 # how many data points we've added
    if agg_cnt == 10000: # stop adding data points when done
        break


# In[9]:


# 100 epochs, Adam optimizer, eta = 0.001. epochs and eta can be tweaked for better performance
eta = 1e-3
num_epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=eta)
for epoch in range(num_epochs):
    tr_loss = 0.0
    print('Epoch {}'.format(epoch))
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        optimizer.zero_grad()  # zero param gradients
        # forward, backward, optimize
        images.unsqueeze(0) # add batch = 1
        print(images.size())
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        tr_loss += loss.item()
        if i % 1999 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / i))
print('Done Training')


# In[ ]:


# save model?
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

