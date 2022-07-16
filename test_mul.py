import torch

t1 = torch.linspace(-1, 1, 12).reshape(2,2,3)
t2 = torch.linspace(-1, 1, 6).reshape(3,2)

t1.requires_grad = True
t2.requires_grad = True

m = t1 @ t2
s = m.sum()

print(t1)
print(t2)
print(m)
print(s)

s.backward()

print(t1.grad)
print(t2.grad)