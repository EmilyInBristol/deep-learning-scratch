import torch # type: ignore
from torch import nn # type: ignore

def batch_norm(X, gamma, beta, moving_mean, moving_var, momentum=0.1, eps=1e-5):
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0,2,3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = (1 - momentum) * moving_mean + momentum * mean
        moving_var = (1 - momentum) * moving_var + momentum * var

    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super.__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, X):
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, 
            self.moving_var, momentum=0.1, eps=1e-5)
        return Y

X = torch.arange(12, dtype=torch.float)
print(X)
X = X.reshape(3, -1)
print(X)

batch_norm(X, 0.01)



