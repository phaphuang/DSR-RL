import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

__all__ = ['DSRRL']

class SelfAttention(nn.Module):

    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)



    def forward(self, x):
        n = x.shape[0]  # sequence length

        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        V = self.V(x)

        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,0))

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)
            logits[torch.eye(n).byte()] = -float("Inf")

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")

        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
        y = self.output_linear(y)

        return y, att_weights_

class DSRRL(nn.Module):
    def __init__(self, in_dim=1024, hid_dim=512, num_layers=1, cell='lstm'):
        super(DSRRL, self).__init__()
        
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True)

        self.fc = nn.Linear(hid_dim*2, 1)

        self.att = SelfAttention(input_size=in_dim, output_size=in_dim)

    def forward(self, x):
        h, _ = self.rnn(x)

        m = x.shape[2] # Feature size
        x = x.view(-1, m)

        att_score, att_weights_ = self.att(x)
        
        out_lay = att_score + h
        p = torch.sigmoid(self.fc(out_lay))

        return p, out_lay, att_score 