import torch
import torch.nn as nn

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, activation  = 'relu', skip = False, skip_in_features = None):
        super(GraphConv, self).__init__()
        self.W = torch.nn.Parameter(torch.DoubleTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W)
        
        self.set_act = False
        if activation == 'relu':
            self.activation = nn.ReLU()
            self.set_act = True
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim = 1)
            self.set_act = True
        else:
            self.set_act = False
            raise ValueError("activations supported are 'relu' and 'softmax'")
            
        self.skip = skip
        if self.skip:
            if skip_in_features == None:
                raise ValueError("pass input feature size of the skip connection")
            self.W_skip = torch.nn.Parameter(torch.DoubleTensor(skip_in_features, out_features)) 
            nn.init.xavier_uniform_(self.W_skip)
        
    def forward(self, A, H_in, H_skip_in = None):
        # A must be an n x n matrix as it is an adjacency matrix
        # H is the input of the node embeddings, shape will n x in_features
        self.A = A
        self.H_in = H_in

        identity = torch.eye(self.A.shape[0]).double()
        # TODO create new tensor method
        identity = identity.cuda()
        A_ = torch.add(self.A, identity)
        D_ = torch.diag(A_.sum(1))
        # since D_ is a diagonal matrix, 
        # its root will be the roots of the diagonal elements on the principle diagonal
        # since A is an adjacency matrix, we are only dealing with positive values 
        # all roots will be real
        D_root_inv = torch.inverse(torch.sqrt(D_))
        A_norm = torch.mm(torch.mm(D_root_inv, A_), D_root_inv)
        # shape of A_norm will be n x n
        
        H_out = torch.mm(torch.mm(A_norm, H_in), self.W)
        # shape of H_out will be n x out_features
        
        if self.skip:
            H_skip_out = torch.mm(H_skip_in, self.W_skip)
            H_out = torch.add(H_out, H_skip_out)
        
        if self.set_act:
            H_out = self.activation(H_out)
            
        return H_out

class HigherOrderGraphConv(nn.Module):
    def __init__(self, in_features, out_features, order, activation  = 'relu', skip = False, skip_in_features = None):
        super(HigherOrderGraphConv, self).__init__()
        if out_features % order != 0:
            raise ValueError("out_features needs to be a multiple of order")
        self.order = order
        self.W_list = nn.ParameterList()
        for i in range(self.order):
            # print(out_features/self.order)
            W_i = nn.Parameter(torch.DoubleTensor(in_features, int(out_features/self.order)))
            nn.init.xavier_uniform_(W_i)    # initial
            self.W_list.append(W_i)
        
        self.set_act = False
        if activation == 'relu':
            self.activation = nn.ReLU()
            self.set_act = True
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim = 1)
            self.set_act = True
        else:
            self.set_act = False
            raise ValueError("activations supported are 'relu' and 'softmax'")
            
        self.skip = skip
        if self.skip:
            if skip_in_features == None:
                raise ValueError("pass input feature size of the skip connection")
            self.W_skip = torch.nn.Parameter(torch.DoubleTensor(skip_in_features, out_features)) 
            nn.init.xavier_uniform_(self.W_skip)
        
    def forward(self, A, H_in, H_skip_in = None):
        # A must be an n x n matrix as it is an adjacency matrix
        # H is the input of the node embeddings, shape will n x in_features
        self.A = A
        self.H_in = H_in

        identity = torch.eye(self.A.shape[0]).double()
        # TODO create new tensor method
        identity = identity.cuda()
        A_ = torch.add(self.A, identity)
        D_ = torch.diag(A_.sum(1))
        # since D_ is a diagonal matrix, 
        # its root will be the roots of the diagonal elements on the principle diagonal
        # since A is an adjacency matrix, we are only dealing with positive values 
        # all roots will be real
        D_root_inv = torch.inverse(torch.sqrt(D_))
        A_norm = torch.mm(torch.mm(D_root_inv, A_), D_root_inv)
        # shape of A_norm will be n x n
        
        for i in range(self.order):
            if i==0:
                A_norm_H = torch.mm(A_norm, H_in)
                H_out = torch.mm(A_norm_H, self.W_list[0])
                # print(H_out.size())
            else:
                A_norm_H = torch.mm(A_norm, A_norm_H)   # left multiply
                H_out = torch.cat((torch.mm(A_norm_H, self.W_list[i]), H_out), dim=1)
                # print(H_out.size())
        # print(H_out.size())

        # shape of H_out will be n x out_features
        
        if self.skip:
            H_skip_out = torch.mm(H_skip_in, self.W_skip)
            H_out = torch.add(H_out, H_skip_out)
        
        if self.set_act:
            H_out = self.activation(H_out)
            
        return H_out