import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def compute_loss(outputs, labels):
    criterion = nn.BCELoss()
    loss = criterion(outputs.float(), labels.float())
    return loss



class TextRNN(nn.Module):
    """文本分类，RNN模型"""
    def __init__(self, config, embedding_matrix):
        super(TextRNN, self).__init__()
        # 三个待输入的数据
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])  #定义词向量
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))                 #获取权重
        self.embedding.weight.requires_grad = config.require_grad                            #对词向量是否更新
        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bidirectional=True)
        #self.rnn = nn.GRU(input_size=128, hidden_size=256, num_layers=2, bidirectional=True)
        self.maxPooling = nn.MaxPool1d(config.max_length)
        self.avgPooling = nn.AvgPool1d(config.max_length)
        self.f1 = nn.Sequential(nn.Linear(256*3, 256*3),
                                nn.Dropout(0.2),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(256*3, 1),
                                nn.Sigmoid())

    def forward(self, x, age, gender):
        x = self.embedding(x)     #bs, length, embedding_size
        x, _ = self.rnn(x)
        x_sum = torch.sum(x, dim=1)
        x = x.permute(0, 2, 1)
        x_max = self.maxPooling(x)
        x_min = self.avgPooling(x)
        x_max = x_max.squeeze()
        x_min = x_min.squeeze()
        x = torch.cat((x_max, x_min, x_sum), dim=1)
        x = self.f1(x)
        x = self.f2(x)
        x = x.squeeze(dim=-1)
        loss = compute_loss(x, gender)
        return x, loss

#
# class TextCNN(nn.Module):
#     def __init__(self, config):
#         super(TextCNN, self).__init__()
#         self.embedding = nn.Embedding(config.word_num, 100)
#         self.conv = nn.Conv1d(100, 256, 5)
#         self.f1 = nn.Sequential(nn.Linear(256 * 596, 128),
#                                 nn.ReLU())
#         self.f2 = nn.Sequential(nn.Linear(128, 10),
#                                 nn.Softmax())
#
#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.detach().numpy()
#         x = np.transpose(x, [0, 2, 1])
#         x = torch.Tensor(x)
#         x = Variable(x)
#         x = self.conv(x)
#         x = x.view(-1, 256 * 596)
#         x = self.f1(x)
#         return self.f2(x)

# #test
#
# if __name__ == "__main__":
#     x = torch.randint(0, 50, (30, 50))
#     label = torch.randint(0, 2, (30,1))
#     model = TextRNN(None, None)
#     prob, loss = model(x, None, label)
#     print(prob)