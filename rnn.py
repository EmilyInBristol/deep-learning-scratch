import torch # type: ignore
from torch import nn # type: ignore
import matplotlib.pyplot as plt # type: ignore
from model import train_model, LinearRegression, train_loop, draw
from torch.utils.data import TensorDataset, DataLoader # type: ignore

class Data(nn.Module):
    def __init__(self, T=1000, num_train=600, tau=4):
        self.time = torch.arange(1, T+1, dtype=torch.float32)
        self.x = torch.sin(0.01 * self.time) + torch.randn(T) * 0.2
        self.tau = tau
        self.T = T
        self.num_train = num_train

    def get_dataloader(self, train=True, batch_size=32):
        features = [self.x[i: self.T-self.tau+i] for i in range(self.tau)]
        self.features = torch.stack(features, 1)
        self.labels = self.x[self.tau:].reshape(-1, 1)
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        #return self.features[i], self.labels[i]
        dataset = TensorDataset(self.features[i], self.labels[i])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
        return dataloader

data = Data(T=1000)
train_loader = data.get_dataloader()
test_loader = data.get_dataloader(train=False)
model = LinearRegression(0.01)
#train_model(model, train_x, train_y)
criterion=nn.MSELoss()
optimizer = model.configure_optimizers()
(train_losses, val_losses, val_correct) = train_loop(model, train_loader, test_loader, criterion, optimizer, )
draw(train_losses, val_losses)





"""
plt.plot(data.time, data.x)
plt.xlabel('Time')
plt.ylabel('X')
plt.legend()
plt.show()
"""