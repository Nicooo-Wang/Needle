import numpy as np
import math
import sys
sys.path.append('./python')
np.random.seed(0)
from tqdm import tqdm
import needle.nn as nn
import needle as ndl

MY_DEVICE = ndl.cuda()

def ResidualBlock(in_channels, out_channels, kernel_size, stride, device=None):
    main_path = nn.Sequential(nn.ConvBN(in_channels, out_channels, kernel_size, stride, device),
                              nn.ConvBN(in_channels, out_channels, kernel_size, stride, device))
    return nn.Residual(main_path)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.model = nn.Sequential(nn.ConvBN(3, 16, 7, 4, device=device),
                                   nn.ConvBN(16, 32, 3, 2, device=device),
                                   ResidualBlock(32, 32, 3, 1, device=device),
                                   nn.ConvBN(32, 64, 3, 2, device=device),
                                   nn.ConvBN(64, 128, 3, 2, device=device),
                                   ResidualBlock(128, 128, 3, 1,
                                                 device=device),
                                   nn.Flatten(),
                                   nn.Linear(128, 128, device=device),
                                   nn.ReLU(),
                                   nn.Linear(128, 10, device=device))
        # END YOUR SOLUTION

    def forward(self, x):
        # BEGIN YOUR SOLUTION
        return self.model(x)
        # END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    tot_loss, tot_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:
        model.eval()
        for X, y in tqdm(dataloader, desc="Evaluating", leave=False):
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
    else:
        model.train()
        for X, y in tqdm(dataloader, desc="Training", leave=False):
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    sample_nums = len(dataloader.dataset)
    return tot_error/sample_nums, np.mean(tot_loss)


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    print(data_dir)
    np.random.seed(4)
    resnet = ResNet9(device=MY_DEVICE)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    test_set = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, device=MY_DEVICE)
    test_loader = ndl.data.DataLoader(test_set, batch_size=batch_size, device=MY_DEVICE)

    for epoch_num in range(epochs):
        print(f"Epoch {epoch_num + 1}/{epochs}")
        train_err, train_loss = epoch(train_loader, resnet, opt)
        print(f"Train Error: {train_err * 100:.4f}%, Train Loss: {train_loss * 100:.4f}%")

    test_err, test_loss = epoch(test_loader, resnet, None)
    print(f"Test Error: {test_err*100:.4f}%, Test Loss: {test_loss*100:.4f}%")

    return train_err, train_loss, test_err, test_loss


if __name__ == "__main__":
    train_err, train_loss, test_err, test_loss = train_mnist(
        data_dir="/home/hyjing/Code/DeepLearningSystem/HW4/data/", epochs=50)
