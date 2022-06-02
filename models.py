import torch
import torch.nn as nn

from layer import GraphConv, HigherOrderGraphConv

class GCN_2layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, skip = False):
        super(GCN_2layer, self).__init__()
        self.skip = skip
        
        self.gcl1 = GraphConv(in_features, hidden_features)
        
        if self.skip:
            self.gcl_skip = GraphConv(hidden_features, out_features, activation = 'softmax', skip = self.skip,
                                  skip_in_features = in_features)
        else:
            self.gcl2 = GraphConv(hidden_features, out_features, activation = 'softmax')   # out_features为num_classes
        
    def forward(self, A, X):
        out = self.gcl1(A, X)
        if self.skip:
            out = self.gcl_skip(A, out, X)
        else:
            out = self.gcl2(A, out)
            
        return out

class HigherOrderGCN_2layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, nums_class, order, skip = False):
        super(HigherOrderGCN_2layer, self).__init__()
        self.skip = skip
        
        self.gcl1 = HigherOrderGraphConv(in_features, hidden_features, order)
        
        if self.skip:
            self.gcl_skip = HigherOrderGraphConv(hidden_features, out_features, order, activation = 'relu', skip = self.skip,
                                  skip_in_features = in_features)
        else:
            self.gcl2 = HigherOrderGraphConv(hidden_features, out_features, order, activation = 'relu')   # out_features为num_classes
        self.fully_connected = torch.nn.Linear(out_features, nums_class)


    def forward(self, A, X):
        out = self.gcl1(A, X)
        if self.skip:
            out = self.gcl_skip(A, out, X)
        else:
            out = self.gcl2(A, out)
        out = self.fully_connected(out)   # fully connected
        out = torch.nn.functional.softmax(out, dim=1)
        return out