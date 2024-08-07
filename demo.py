import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import datasets, transforms # type: ignore
import matplotlib.pyplot as plt # type: ignore

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

label = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
def load_fashion_mnist(resize=(28, 28)):
	# define a tranformation to normalize the data
	transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
	# load the dataset with the transformation
	train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
	test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

	print(len(train_dataset))
	print(len(test_dataset))

	sample_image, sample_label = train_dataset[0]
	print(sample_image.shape)
	print(sample_label)
	print(label[sample_label])

	# Convert tensor back to PIL image for visualization
	to_pil = transforms.ToPILImage()
	resized_pil_image = to_pil(sample_image)

	# Display the resized image
	plt.imshow(resized_pil_image)
	plt.title('Resized Image')
	plt.show()

load_fashion_mnist()