import torch # type: ignore
from torch import nn # type: ignore
from model import FashionMNIST, train_loop
import torch.optim as optim # type: ignore
import matplotlib.pyplot as plt # type: ignore
from torch.nn import functional as F # type: ignore
import logging
logging.basicConfig(level=logging.INFO)

# set the device to cpu
device = torch.device('cpu')

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

def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_0 = K.shape[0]

    X = X.reshape((c_i, h*w))
    K = K.reshape((c_0, c_i))

    Y = torch.matmul(K, X)
    Y = Y.reshape((c_0, h, w))
    return Y

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

    def _initialize_weights(self):
        for module in self.net:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X):
        return self.net(X)
    
class BNLeNet(nn.Module):
    def __init__(self, lr=0.1, num_classes=10):
        super(BNLeNet, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.LazyBatchNorm2d(),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.LazyBatchNorm1d(),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.LazyLinear(84), nn.LazyBatchNorm1d(),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.LazyLinear(num_classes)
        )

    def _initialize_weights(self):
        for module in self.net:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X):
        return self.net(X)

    
class AlexNet(nn.Module):
    def __init__(self, lr=0.1, num_classes=10):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1), 
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )

    def forward(self, X):
        return self.net(X)
    
class Residual(nn.Module):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)

        Y += X
        return F.relu(Y)

        

def layer_summary(X_shape, model):
    X = torch.randn(X_shape)
    for layer in model.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

def draw(train_losses, val_losses, val_correct, num_epochs=5):
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, num_epochs), train_losses, label='Train Loss')
    plt.plot(range(0, num_epochs), val_losses, label='Validation Loss', linestyle='--')
    plt.plot(range(0, num_epochs), val_correct, label='Validation Correction')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    resize = (224, 224)
    batch_size = 128
    #model = LeNet(lr=0.1).to(device)
    #model = AlexNet(lr=0.1)
    model = BNLeNet(lr=0.1).to(device)
    #X_shape = (1, 1) + resize
    #layer_summary(X_shape, model)

    dummy_input = torch.randn(2, 1, 224, 224)
    model(dummy_input)
    model._initialize_weights()
    
    fashion_mnist = FashionMNIST(batch_size, resize)
    train_loader = fashion_mnist.get_dataloader(True)
    val_loader = fashion_mnist.get_dataloader(False)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_losses, val_losses, val_correct = train_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs=3)
    draw(train_losses, val_losses, val_correct, num_epochs=3)

    

    
    


