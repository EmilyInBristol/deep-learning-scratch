import torch
import torch.nn as nn

class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()
		self.linear = nn.Linear(10, 1)

	def forward(self, x):
		return self.linear(x)

	def sgd(self, learning_rate):
		with torch.no_grad():
			for param in self.parameters():
				if param.grad is not None:
					param -= param.grad * learning_rate


x = torch.arange(10, dtype=torch.float32)
x = x.view(1, -1)
y = torch.tensor([[20.0]])

m = MyModel();

criterion = nn.MSELoss()
y_hat = m.forward(x)
loss = criterion(y_hat, y)

print(y_hat, y)
print(loss)

parameters = list(m.parameters())
print(parameters)
loss.backward()

gradients = [p.grad for p in parameters]

m.sgd(0.01)

print(list(m.parameters()))
