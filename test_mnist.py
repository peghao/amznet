'''
test mnist
'''
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

def softmax(X):
    exp_x = torch.exp(X)
    return exp_x/torch.sum(exp_x, dim=-1).reshape(X.shape[0], 1)

def MYLoss(Y_hat, Y):
    return -(torch.log(Y_hat) * Y).sum()

BATCHSIZE = 128

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.MNIST("../dataset", train=True, transform=transform)

W1 = torch.linspace(-0.05, 0.05, 28*28*100).reshape(28*28, 100)
W2 = torch.linspace(-0.05, 0.05, 100*100).reshape(100, 100)
W3 = torch.linspace(-0.05, 0.05, 100*10).reshape(100, 10)
b1 = torch.linspace(-0.05, 0.05, 100)
b2 = torch.linspace(-0.05, 0.05, 100)
b3 = torch.linspace(-0.05, 0.05, 10)

W1.requires_grad = True
W2.requires_grad = True
W3.requires_grad = True
b1.requires_grad = True
b2.requires_grad = True
b3.requires_grad = True

X0 = train_data.data[0:5].float().reshape(5, 28*28)
Y = torch.nn.functional.one_hot(train_data.targets[0:5], num_classes=10)

for steps in range(100):

    X1 = torch.relu(X0 @ W1 + b1)
    X2 = torch.relu(X1 @ W2 + b2)
    X3 = X2 @ W3 + b3

    Y_hat = softmax(X3)
    loss = MYLoss(Y_hat, Y)
    loss.backward()

    print(f"steps:{steps}, loss:{loss/5}");

    lr = 0.00001
    with torch.no_grad():
        W1 -= lr*W1.grad
        W2 -= lr*W2.grad
        W3 -= lr*W3.grad
        b1 -= lr*b1.grad
        b2 -= lr*b2.grad
        b3 -= lr*b3.grad