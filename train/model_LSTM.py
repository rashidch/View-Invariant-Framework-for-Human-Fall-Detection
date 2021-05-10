import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class FallNet(nn.Module):
    def __init__(self, input_dim=24, class_num=2):
        super(FallNet, self).__init__()
        self.input = input_dim
        self.num_layers = 3
        self.hidden_state = 24
        self.lstm = nn.LSTM(self.input, self.hidden_state, self.num_layers, batch_first=True, dropout=0.5)
        self.linear = nn.Sequential(nn.Linear(self.hidden_state, 2), nn.ELU())
        self.class_num = class_num

    def forward(self, inputs):
        features, _ = self.lstm(inputs)
        raw_preds = self.linear(features[:, -1, :])
        output_probs = F.softmax(raw_preds, dim=1)
        return raw_preds, output_probs


class GenNet(nn.Module):
    def __init__(self, Num):
        super(GenNet, self).__init__()

        self.num_layers = 3
        self.hidden_state = Num
        self.Enlstm = nn.LSTM(24, self.hidden_state, 2, batch_first=True, dropout=0.5)
        self.Delstm = nn.LSTM(self.hidden_state, 24, 2, batch_first=True, dropout=0.5)

    def forward(self, inputs):
        encoder, _ = self.Enlstm(inputs)
        decoder, _ = self.Delstm(encoder)
        return decoder
