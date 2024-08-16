import torch
from torch import nn

def cpu():
    return torch.device('cpu')

def gpu(i=0):  #@save
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

def num_gpus():
    return torch.cuda.device_count()

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

class conv2D(nn.Module):
    def __init__(self, kernal_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernal_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    
def demo_conv():
    X = torch.ones((6, 8))
    X[:,2:6] = 0
    kernal = torch.tensor([[1.0, -1.0]])

    Y = corr2d(X, kernal)

    epoch = 10
    model = nn.LazyConv2d(1, kernel_size=(1,2), bias=False)

    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    for i in range(epoch):
        #model = conv2D(kernal.shape)

        y_hat = model(X)

        loss = (y_hat - Y) ** 2
        loss = loss.sum()
        model.zero_grad()

        loss.backward()
        model.weight.data[:] -= model.weight.grad * 0.03

        if i % 2 == 0:
            print(f'i:{i} loss:{loss}')

    print(model.weight.data.reshape(1,2))


def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape) # add exmpale and channel dimention
    Y = conv2d(X)
    Y = Y.reshape(Y.shape[2:])

"""
conv2d = nn.LazyConv2d(1, padding=1, kernel_size=3)
X = torch.rand(size = (8,8))
ret = comp_conv2d(conv2d, X)
#print(ret.shape)
"""

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

Y = sum(corr2d(x, k) for x, k in zip(X, K))
print(Y)


    
    


