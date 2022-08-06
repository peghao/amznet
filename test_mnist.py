'''
test mnist
'''
from typing import List
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
class Model(ABC):
    # _params = []

    @abstractmethod
    def forward(self, X:torch.Tensor):
        pass

    @abstractmethod
    def params(self):
        pass

class Linear(Model):

    def __init__(self, input_features, output_features) -> None:
        super().__init__()
        self.__params = []
        self.__params.append(torch.linspace(-0.05, 0.05, input_features*output_features, dtype=torch.float32).reshape(input_features, output_features))
        self.__params.append(torch.linspace(-0.05, 0.05, output_features, dtype=torch.float32).reshape(1, output_features))
        self.__params[0].requires_grad = True
        self.__params[1].requires_grad = True
        with torch.no_grad():
            self.__params[0].grad = torch.zeros_like(self.__params[0])
            self.__params[1].grad = torch.zeros_like(self.__params[1])

    def forward(self, X:torch.Tensor):
        W = self.__params[0]
        b = self.__params[1]
        return X@W + b

    def params(self):
        super().params()
        return self.__params

class FCNet(Model):

    def __init__(self) -> None:
        super().__init__()
        self.model1 = Linear(28*28, 100)
        self.model2 = Linear(100, 100)
        self.model3 = Linear(100, 10)

    def forward(self, X:torch.Tensor):
        x1 = torch.relu(self.model1.forward(X))
        x2 = torch.relu(self.model2.forward(x1))
        x3 = self.model3.forward(x2)
        return x3

    def params(self):
        super().params()
        W = []
        W+=self.model1.params()
        W+=self.model2.params()
        W+=self.model3.params()
        return W

class SGD:

    def __init__(self, params:List, lr=0.001, p=0.9) -> None:
        self.lr = lr
        self.p = p
        self.params = params
        self.v = [torch.zeros_like(x) for x in params]

    def step(self):
        with torch.no_grad():
            for i in range(len(self.params)):
                self.v[i] = self.p*self.v[i] - self.lr*self.params[i].grad
                self.params[i] += self.v[i]

    def zero_grad(self):
        with torch.no_grad():
            for x in self.params:
                x.grad *= 0.0

def softmax(X):
    exp_x = torch.exp(X)
    return exp_x/torch.sum(exp_x, dim=-1).reshape(X.shape[0], 1)

def MYLoss(Y_hat, Y):
    return -(torch.log(Y_hat) * Y).sum()

'''图片归一化'''
def norm(X):
    return (X - torch.min(X))/(torch.max(X)- torch.min(X))

def norm_batched(X):
    for i in range(len(X)):
        X[i] = norm(X[i])

BATCHSIZE = 128

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.MNIST("../dataset", train=True, transform=transform)

net = FCNet()
opt = SGD(net.params())

num_batch = 400
for epoch in range(5):
    for i in range(num_batch):
        
        X0 = train_data.data[i*BATCHSIZE:(i+1)*BATCHSIZE].float().reshape(BATCHSIZE, 28*28)
        Y = torch.nn.functional.one_hot(train_data.targets[i*BATCHSIZE:(i+1)*BATCHSIZE], num_classes=10)

        norm_batched(X0)
        
        net_out = net.forward(X0)

        Y_hat = softmax(net_out)
        loss = MYLoss(Y_hat, Y)

        print(f"epoch:{epoch}, loss:{loss/BATCHSIZE}, prograss:{i}/{num_batch}");
        
        opt.zero_grad()
        loss.backward()
        opt.step()        