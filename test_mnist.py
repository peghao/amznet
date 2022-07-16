'''
test mnist
'''
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

BATCHSIZE = 128

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.MNIST("../../dataset", train=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCHSIZE, shuffle=False, num_workers=1)
# plt.imshow(train_data.data[0], cmap='gray')

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
X1 = torch.relu(X0 @ W1 + b1)
X2 = torch.relu(X1 @ W2 + b2)
X3 = X2 @ W3 + b3
X3.retain_grad()

def softmax(X):
    exp_x = torch.exp(X)
    return exp_x/torch.sum(exp_x, dim=-1).reshape(X.shape[0], 1)

def MYLoss(Y_hat, Y):
    return -(torch.log(Y_hat) * Y).sum()

Y_hat = softmax(X3)
loss = MYLoss(Y_hat, Y)
loss.backward()

# print(train_data.targets[0:5])
print(loss)
# print(X3)
# print(X1)
print(b1.grad)
print(b2.grad)
print(b3.grad)
# print(W2.grad)