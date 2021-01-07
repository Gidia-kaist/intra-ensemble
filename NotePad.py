import torch
import torch.nn as nn


mask1 = torch.bernoulli(0.3 * torch.ones(100, 100))
print(mask1.sum()/10000)