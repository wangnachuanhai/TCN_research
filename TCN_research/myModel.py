import torch
import torch.nn as nn
from tcn import TemporalConvNet as TCN


class mytcn(nn.Module):
    def __init__(self, num_inputs, num_channels, dropout, batch, kernel_size=3):
        super(mytcn, self).__init__()
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.drop = dropout
        self.batch = batch

        self.modelname = 'tcn'

        self.tcn = TCN(self.num_inputs, self.num_channels, self.kernel_size, self.drop)
        self.fc = nn.Linear(17*409, 17)
        self.drop = nn.Dropout(self.drop)

    def forward(self, x):
        x = self.tcn(x)
        # print(x.shape)
        x = x.view(self.batch, -1)
        x = self.fc(x)
        x = self.drop(x)
        return x