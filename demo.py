import torch # type: ignore
import torch.nn as nn # type: ignore

class SynthesisData():
	def __init__(self, w, b, num_train=2, num_test=1, noise=0.01):
		n = num_test + num_train
		self.x = torch.randn(n, len(w))
		noise = torch.randn(n, 1) * noise
		self.y = torch.matmul(self.x, w.reshape(-1, 1)) + b + noise

def generate_data():
	w = torch.tensor([2, -3.4])
	b = 4.2
	data = SynthesisData(w, b, 10, 2)
	return data.x[:10], data.y[:10], data.x[10:], data.y[10:]

class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()
		self.linear = nn.Linear(2, 1)

	def forward(self, x):
		return self.linear(x)

	def sgd(self, learning_rate):
		with torch.no_grad():
			for param in self.parameters():
				if param.grad is not None:
					param -= param.grad * learning_rate


def train_model(model, train_x, train_y, num_epochs=100, learning_rate=0.01):
	criterion = nn.MSELoss()

	for epoch in range(num_epochs):
		model.train()
		y_hat = model.forward(train_x)
		loss = criterion(y_hat, train_y)
		# zero gradients
		model.zero_grad()
		# calculate gradient
		loss.backward()
		# update parameters
		model.sgd(learning_rate)

		if epoch % 10 == 0:
			print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}')

m = MyModel()
train_x, train_y, test_x, test_y = generate_data()
train_model(m, train_x, train_y)

print(m.linear.weight)
print(m.linear.bias)