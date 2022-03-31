import torch
import torch.nn as nn


a = torch.randn(3, 5, 10)
print(a)
b = nn.MaxPool2d((5, 1))
c = b(a)

