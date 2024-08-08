import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import datasets, transforms # type: ignore
import matplotlib.pyplot as plt # type: ignore
from torch.utils.data import DataLoader # type: ignore
import torch.optim as optim # type: ignore


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

def visualize(train_loader):
	#print(len(train_loader), len(val_loader))
	X, y = next(iter(train_loader))
	#print(X.shape, X.dtype, y.shape, y.dtype)

	# Convert tensor back to PIL image for visualization
	to_pil = transforms.ToPILImage()
	resized_pil_image = to_pil(X[0])

	# Display the resized image
	plt.imshow(resized_pil_image)
	plt.title(f'{FASHION_MNIST_LABELS[y[0]]}')
	plt.show()



def softmax(X):
	X_exp = torch.exp(X)
	partition = X_exp.sum(1, keepdims=True)
	return X_exp/partition

class SoftmaxRegressionScratch():
	def __init__(self, num_inputs, num_outputs) -> None:
		self.W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
		self.b = torch.zeros(num_outputs, requires_grad=True)

	def parameters(self):
		return [self.W, self.b]
	
	def forward(self, X):
		X = X.reshape(-1, self.W.shape[0])
		return softmax(torch.matmul(X, self.W) + self.b)


def train_loop():
	resize = (28, 28)

	fashion_mnist = FashionMNIST(64, resize)
	train_loader = fashion_mnist.get_dataloader(train=True)
	val_loader = fashion_mnist.get_dataloader(train=False)

	num_inputs = resize[0] * resize[1]
	num_outputs = 10
	model = SoftmaxRegressionScratch(num_inputs, num_outputs)

	criterion = nn.CrossEntropyLoss() 
	optimizer = optim.SGD(model.parameters(), lr=0.01)

	num_epochs = 5
	for epoch in range(num_epochs):
		total_loss = 0
		correct = 0
		total = 0
		for X, y in train_loader:


			y_hat = model.forward(X)
			loss = criterion(y_hat, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Accumulate loss and accuracy
			total_loss += loss.item()
			_, predicted = torch.max(y_hat, 1)
			total += y.size(0)
			correct += (predicted == y).sum().item()

		# Print epoch statistics
		print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

	print("Training complete.")

train_loop()