import torch # type: ignore
from torch import nn # type: ignore
import matplotlib.pyplot as plt # type: ignore

class Data(nn.Module):
    def __init__(self, T=10, num_train=60):
        self.time = torch.arange(1, T+1, dtype=torch.float32)
        self.x = torch.sin(0.01 * self.time) + torch.randn(T) * 0.2


data = Data(T=1000)

plt.plot(data.time, data.x)
plt.xlabel('Time')
plt.ylabel('X')
plt.legend()
plt.show()