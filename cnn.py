import torch
from torch import nn
from model import FashionMNIST, train_loop
import torch.optim as optim # type: ignore
import logging
logging.basicConfig(level=logging.DEBUG)

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

def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of K, and each time, perform
    # cross-correlation operations with input X. All of the results are
    # stacked together
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

"""
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

Y = corr2d_multi_in(X, K)
print(Y)
"""

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_0 = K.shape[0]

    X = X.reshape((c_i, h*w))
    K = K.reshape((c_0, c_i))

    Y = torch.matmul(K, X)
    Y = Y.reshape((c_0, h, w))
    return Y

"""
X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(Y1, Y2)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
"""

class LeNet(nn.Module):
    def __init__(self, lr=0.1, num_classes=10):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, X):
        return self.net(X)

def layer_summary(X_shape):
    X = torch.randn(X_shape)
    for layer in model.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)


if __name__ == '__main__':

    #layer_summary((1, 1, 28, 28))

    resize = (28, 28)
    batch_size = 64
    model = LeNet(lr=0.1)
    fashion_mnist = FashionMNIST(batch_size)
    train_loader = fashion_mnist.get_dataloader(True)
    val_loader = fashion_mnist.get_dataloader(False)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)




    
    


