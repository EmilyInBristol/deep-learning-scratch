import torch # type: ignore
from torch import nn # type: ignore
import matplotlib.pyplot as plt # type: ignore
from model import train_model, LinearRegression, train_loop, plot_predictions
from torch.utils.data import TensorDataset, DataLoader # type: ignore
import collections
from torch.nn import functional as F # type: ignore
import torch.optim as optim # type: ignore

from kaggle import get_data_frame

import os
import re
import logging
logging.basicConfig(level=logging.INFO)

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
    
def test_k_step_ahead():
    data = Data(T=1000)
    train_loader = data.get_dataloader()
    test_loader = data.get_dataloader(train=False)
    model = LinearRegression(0.01)
    criterion=nn.MSELoss()
    optimizer = model.configure_optimizers()
    (train_losses, val_losses, val_correct) = train_loop(model, train_loader, test_loader, criterion, optimizer, )
    onestep_preds = model(data.features).detach().numpy()
    """
    plot_predictions(data.time[data.tau:], data.labels, onestep_preds, 'time', 'x', legend=['labels', '1-step preds'], figsize=(6, 3))
    #draw(train_losses, val_losses)
    """
    multistep_preds = torch.zeros(data.T)
    multistep_preds[:] = data.x
    for i in range(data.num_train+data.tau, data.T):
        multistep_preds[i] = model(multistep_preds[i-data.tau:i].reshape(1, -1))

    print(multistep_preds[:10])
    multistep_preds = multistep_preds.detach().numpy()
    print(multistep_preds[:10])

    plot_predictions([data.time[data.tau:], data.time[data.num_train+data.tau:]], [onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time','x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))

class TimeMachine(nn.Module):
    def __init__(self, num_steps, batch_size, num_train=10000, num_val=5000):
        super().__init__()
        corpus, self.vocab = self.build(self._download())
        array = torch.tensor([corpus[i:i+num_steps+1] for i in range(len(corpus)-num_steps)])
        self.X, self.Y = array[:,:-1], array[:,1:]

        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size

    def _download(self):
        DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
        sha1_hash = '090b5e7e70c295757f55df93cb0a180b9691891a'
        folder = './data/timemachine'
        raw_text = get_data_frame(DATA_URL, sha1_hash, folder)
        return raw_text

    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', ' ', text)
    
    def _tokenize(self, text):
        return list(text)
    
    def build(self, raw_text, vocab=None):
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab
    
    def get_dataloader(self, train=True):
        i = slice(0, self.num_train) if train else slice(
            self.num_train, self.num_train + self.num_val)
        dataset = TensorDataset(self.X[i], self.Y[i])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=train)
        return dataloader
    
class Vocab():
    def __init__(self, tokens=[], min_freq=0):
        # flatten a 2d list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        self.idx_to_token = list(sorted(set(['<unk>']+[token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (tuple, list)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(i)] for i in indices]
        return self.idx_to_token[indices]
    
    @property
    def unk(self):
        return self.token_to_idx['<unk>']
    
class RNNScratch(nn.Module):  
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.W_xh = nn.Parameter(
            torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))

        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma

    def forward(self, inputs, state=None):
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = torch.zeros((inputs.shape[1], self.num_hiddens),
                            device=inputs.device)
        else:
            state, = state
        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
            state = torch.tanh(torch.matmul(X, self.W_xh) +
                            torch.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state
    
class RNNLMScratch(nn.Module):  
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size):
        super().__init__()
        self.rnn = rnn
        self.vocab_size = vocab_size
        self.init_params()

    def init_params(self):
        self.W_hq = nn.Parameter(
            torch.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))
    
    def loss(self, Y_hat, Y, averaged=True):
        #Y_hat.shape = torch.Size([1024, 16, 53]) Reshape: [16384, 53]
        Y_hat = torch.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        #Y = torch.Size([1024, 16]) Flattened Dimension: 1024 * 16 = 16384
        Y = torch.reshape(Y, (-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

    def one_hot(self, X):
        # Output shape: (num_steps, batch_size, vocab_size)
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)

    def training_step(self, batch):
        #print(self(*batch[:-1]).shape, batch[-1].shape)
        l = self.loss(self(*batch[:-1]), batch[-1])
        print(torch.exp(l).item())
        #self.plot('ppl', torch.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        #self.plot('ppl', torch.exp(l), train=False)
        return l
        
    def output_layer(self, rnn_outputs):
        outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return torch.stack(outputs, 1) #[batch_size, num_steps, vocab_size].

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)

    def predict(self, prefix, num_preds, vocab):
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = torch.tensor([[outputs[-1]]])
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1:  # Warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else:  # Predict num_preds steps
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(Y.argmax(axis=2).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])
    
class Trainer():
    def __init__(self) -> None:
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.train_ppls = []
        self.val_ppls = []

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def fit_epoch(self, model, train_dataloader, val_dataloader, optimizer, gradient_clip_val=0, epoch_num=100):

        for epoch in range(epoch_num):
            model.train()
            for batch in train_dataloader:
                loss = model.training_step(batch)
                optimizer.zero_grad()
                loss.backward()
                if gradient_clip_val > 0:  # To be discussed later
                    self.clip_gradients(gradient_clip_val, model)
                optimizer.step()
                self.train_batch_idx += 1
                self.train_ppls.append(torch.exp(loss).item())


            if val_dataloader is None:
                return
            
            model.eval()
            for batch in val_dataloader:
                loss = model.validation_step(batch)
                self.val_batch_idx += 1
                self.val_ppls.append(torch.exp(loss).item())

def draw(train_loss, val_loss):
	# Plotting the losses
	plt.figure(figsize=(10, 6))
	plt.plot(range(0, len(train_loss)), train_loss, label='Train PPL')
	plt.plot(range(1, len(val_loss)+1), val_loss, label='Val PPL', linestyle='--')
	plt.xlabel('Epochs')
	plt.ylabel('PPL')
	plt.title('Training and Validation PPL Over Epochs')
	plt.legend()
	plt.show()
        
if __name__ == '__main__':

    data = TimeMachine(batch_size=1024, num_steps=16)
    #for X, Y in data.get_dataloader():
    #    print('X:', X, '\nY:', Y)
    #    break
    train_loader = data.get_dataloader()
    val_loader = data.get_dataloader(train=False)
    
    #batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    #X = torch.ones((num_steps, batch_size, num_inputs)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab))
    #X = torch.ones((batch_size, num_steps), dtype=torch.int64)
    #outputs = model(X)
    #print(outputs.shape)
    optimizer = optim.SGD(model.parameters(), lr=1)

    trainer = Trainer()
    trainer.fit_epoch(model, train_loader, val_loader, optimizer, gradient_clip_val=1, epoch_num=100)
    draw(trainer.train_ppls, trainer.val_ppls)
    print(model.predict('it has', 20, data.vocab))



