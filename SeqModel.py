import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def compute_loss(outputs, labels):
    loss = F.cross_entropy(outputs, labels)
    return loss



class TextRNN(nn.Module):
    """文本分类，RNN模型"""
    def __init__(self, config, embedding_matrix):
        super(TextRNN, self).__init__()
        # 三个待输入的数据
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = config.require_grad
        self.rnn = nn.LSTM(input_size=100, hidden_size=128, num_layers=2, bidirectional=True)
        #self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        self.f1 = nn.Sequential(nn.Linear(256, 128),
                                nn.Dropout(0.2),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, 1),
                                nn.Sigmoid())

    def forward(self, x, age, gender):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.f1(x[:, -1, :])
        loss = compute_loss(x, gender)
        return self.f2(x), loss


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.word_num, 100)
        self.conv = nn.Conv1d(100, 256, 5)
        self.f1 = nn.Sequential(nn.Linear(256 * 596, 128),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, 10),
                                nn.Softmax())

    def forward(self, x):
        x = self.embedding(x)
        x = x.detach().numpy()
        x = np.transpose(x, [0, 2, 1])
        x = torch.Tensor(x)
        x = Variable(x)
        x = self.conv(x)
        x = x.view(-1, 256 * 596)
        x = self.f1(x)
        return self.f2(x)