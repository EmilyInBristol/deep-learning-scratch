import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import datasets, transforms # type: ignore
import matplotlib.pyplot as plt # type: ignore
from torch.utils.data import DataLoader # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import matplotlib.pyplot as plt # type: ignore


##############linear regression with no active function###############
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
		# add weight decay, l2 regularization
		l2_lambda = 0.01
		l2_reg = torch.tensor(0.)
		for name, param in model.named_parameters():
			if 'bias' not in name: # excludes biases
				l2_reg += torch.norm(param)
		loss = loss + l2_lambda * l2_reg
		# zero gradients
		model.zero_grad()
		# calculate gradient
		loss.backward()
		# update parameters
		model.sgd(learning_rate)

		if epoch % 10 == 0:
			print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}')

def demo_linear_model():
	m = MyModel()
	train_x, train_y, test_x, test_y = generate_data()
	train_model(m, train_x, train_y)

	print(m.linear.weight)
	print(m.linear.bias)

#############classification model##############

FASHION_MNIST_LABELS = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

class FashionMNIST():
	def __init__(self, batch_size, resize=(28, 28)) -> None:
		self.batch_size = batch_size
		transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
		self.train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
		self.val = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

	def get_dataloader(self, train):
		data = self.train if train else self.val
		return DataLoader(data, self.batch_size, shuffle=train, num_workers=0)

class MLP(nn.Module):
	def __init__(self, num_inputs, num_hidden, num_outputs, dropout_prob=0.5):
		super(MLP, self).__init__()
		self.dropout_prob = dropout_prob
		self.net = nn.Sequential(nn.Linear(num_inputs, num_hidden),
						   nn.ReLU(),
						   nn.Dropout(dropout_prob),
						   nn.Linear(num_hidden, num_outputs))
		
		self._initialize_weights()
	
	def forward(self, X):
		X = X.reshape(X.size(0), -1)
		return self.net(X)
	
	def _initialize_weights(self):
		for layer in self.net:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)
				if layer.bias is not None:
					nn.init.zeros_(layer.bias)

def train_loop():
	resize = (28, 28)

	fashion_mnist = FashionMNIST(64, resize)
	train_loader = fashion_mnist.get_dataloader(train=True)
	val_loader = fashion_mnist.get_dataloader(train=False)

	num_inputs = resize[0] * resize[1]
	num_hidden = 256
	num_outputs = 10
	model = MLP(num_inputs, num_hidden, num_outputs)

	criterion = nn.CrossEntropyLoss() 
	optimizer = optim.SGD(model.parameters(), lr=0.01)

	num_epochs = 5
	train_losses = []
	val_losses = []
	for epoch in range(num_epochs):

		total_train_loss = 0
		correct_train = 0
		total_train = 0
		model.train()  # Set the model to training mode
		for X, y in train_loader:
			y_hat = model.forward(X)
			loss = criterion(y_hat, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Accumulate loss and accuracy
			total_train_loss += loss.item()
			_, predicted = torch.max(y_hat, 1)
			total_train += y.size(0)
			correct_train += (predicted == y).sum().item()

		# Print epoch statistics
		#print(f"Epoch [{epoch+1}/{num_epochs}], TrainingLoss: {total_train_loss/len(train_loader):.4f}, Accuracy: {100 * correct_train / total_train:.2f}%")

		total_val_loss = 0
		total_val = 0
		correct_val = 0
		val_looses = []
		model.eval()  # Set the model to training mode
		with torch.no_grad():
			for X, y in val_loader:
				y_hat = model.forward(X)
				loss = criterion(y_hat, y)

				# Accumulate loss and accuracy
				total_val_loss += loss.item()
				_, predicted = torch.max(y_hat, 1)
				total_val += y.size(0)
				correct_val += (predicted == y).sum().item()


		avg_train_loss = total_train_loss/len(train_loader)
		avg_val_loss = total_val_loss/len(val_loader)
		train_losses.append(avg_train_loss)
		val_losses.append(avg_val_loss)

		print(f"""Epoch [{epoch+1}/{num_epochs}], 
	TrainLoss: {avg_train_loss:.4f}, 
	ValidationLoss: {avg_val_loss:.4f}, 
	ValidationAccuracy: {100 * correct_val / total_val:.2f}%""")


	# Plotting the losses
	plt.figure(figsize=(10, 6))
	plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
	plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Training and Validation Loss Over Epochs')
	plt.legend()
	plt.show()


train_loop()


def dropout_layer(X, dropout):
	assert 0 <= dropout <= 1
	if dropout == 1: return torch.zeros_like(X)
	mask = (torch.rand(X.shape) > dropout).float()
	print(mask)
	return (X * mask) / (1-dropout)

def test_dropout_layer():
	X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
	print(X)
	print(X.mean().item())
	a = dropout_layer(X, 0.5)
	print(a.mean().item())

