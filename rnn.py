import torch # type: ignore
from torch import nn # type: ignore

class Data(nn.Module):
    def __init__(self, T=10, num_train=60):
        pass

time = torch.arange(1, 11, dtype=torch.float32)
print(time)