import torch
from torch import nn

def cpu():
    return torch.device('cpu')

def gpu(i=0):  #@save
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

def num_gpus():
    return torch.cuda.device_count()
 

print(cpu())
print(gpu())
print(gpu(1))
print(num_gpus())
x = torch.tensor([1,2,3])
print(x.device)
