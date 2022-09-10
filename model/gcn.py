import math
import torch
import torch.nn as nn

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.init_weights()
    
    def init_weights(self):
        stdv = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj_matrix):
        sp = torch.matmul(x, self.weights)
        output = torch.matmul(adj_matrix, sp)
        if self.bias is not None:
            return output + self.bias
        else:
            return output