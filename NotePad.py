from torchvision import datasets
import torchvision.transforms as transforms
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
z = torch.zeros(512, 512, requires_grad=True)
print(z)
a = nn.Linear(512, 512)
b = a.weight
a.weight = z

if switch_ensemble is True:
    print("Ensemble:", switch_ensemble)
    drop = self.fc1.weight
    mask = np.random.choice([0, 1], size=(512, 784), p=[0.1, 0.9])
    mask = torch.tensor(mask)
    drop = drop.mul(mask)
    self.fc1.weight = torch.nn.Parameter(drop)
    print(self.fc1.weight.sum())
# linear layer (n_hidden -> hidden_2)