import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# 1.Basic autograd example 1

#Creates tensors
x = torch.tensor(1., requires_grad = True)
w = torch.tensor(2., requires_grad = True)
b = torch.tensor(3., requires_grad = True)

# Build a computational graph
y = w * x + b 

# Compute gradients
y.backward() # automatic differential

print(x.grad)
print(w.grad)
print(b.grad)

# Basic autograd example 2

x = torch.randn(10, 3)
y = torch.randn(10, 2)

linear = nn.Linear(3, 2)
print('w : ', linear.weight)
print('b : ', linear.bias)

criterion = nn.MSELoss() # Loss function
optimizer = torch.optim.SGD(linear.parameters(), lr = 0.01) # learning rate : 0.01

pred = linear(x) 

loss = criterion(pred, y) # compute loss between pred and y
print('loss : ', loss.item())

loss.backward()

print('dL/dw : ', linear.weight.grad)
print('dL/db : ', linear.bias.grad)

optimizer.step() # update parameters

pred = linear(x) 
loss = criterion(pred, y)
print('loss after 1 step optimization : ', loss.item())