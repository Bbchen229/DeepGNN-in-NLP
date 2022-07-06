#!/usr/bin/env python
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__( self, input_dim, \
                        output_dim, \
                        act_func = None, \
                        featureless = False, \
                        dropout_rate = 0., \
                        bias=False):
        super(GraphConvolution, self).__init__()

        self.featureless = featureless
        setattr(self, 'W', nn.Parameter(torch.randn(input_dim, output_dim)))
        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj):
        x = self.dropout(x)
        if self.featureless:
            pre_sup = getattr(self, 'W')
        else:
            pre_sup = x.mm(getattr(self, 'W'))

        out = adj.mm(pre_sup)
        if self.act_func is not None:
            out = self.act_func(out)
        #for visualization
        self.embedding = out
        return out




class AdaptiveConv(nn.Module):
    def __init__(self, in_features, out_features, variant=False):
        super(AdaptiveConv, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input,adj, h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        return output



class GCN(nn.Module):
    def __init__( self, input_dim, \
                        nlayers,\
                        nhidden = 200, \
                        nclass=10,\
                        dropout_rate=0., \
                        wd = 0.\
                        ):
        super(GCN, self).__init__()
        
        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, nhidden, act_func=nn.ReLU(), featureless=True, dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(nhidden, nclass, dropout_rate=dropout_rate)
        self.convs = nn.ModuleList()
        for _ in range(nlayers-2):
            self.convs.append(GraphConvolution(nhidden, nhidden,act_func=nn.ReLU(),dropout_rate=dropout_rate))
        self.dropout = dropout_rate
        self.act_fn = nn.ReLU()
      

    def forward(self, x, adj):
        out = self.layer1(x,adj)
        for i, con in enumerate(self.convs):
            out = F.dropout(out,self.dropout,training=self.training)
            out = self.act_fn(con(out,adj))
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.layer2(out,adj)
        return out


class DeepGCN(nn.Module):
    def __init__(self, input_dim, \
                 nlayers, \
                 nhidden=200, \
                 nclass=10, \
                 dropout_rate=0.5, \
                 beta=0.1, \
                 alpha=0.1,\
                 var = False,\
                 wd=0.
                 ):
        super(DeepGCN, self).__init__()

        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, nhidden, act_func=nn.ReLU(), featureless=True,
                                       dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(nhidden, nclass, dropout_rate=dropout_rate)

        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(AdaptiveConv(nhidden, nhidden, variant=var))
        self.act_fn = nn.ReLU()
        self.dropout = dropout_rate
        self.alpha = alpha
        self.lamda = beta
        self.num_layers = nlayers
        self.params1 = list(self.convs.parameters())    
        self.params2 = list(self.layer1.parameters())
        self.params3 = list(self.layer2.parameters())


    def forward(self, x,adj):
        layer_inner = self.layer1(x,adj)
        h0 = layer_inner
        layer0 = h0
        for i, con in enumerate(self.convs):
            if i >= 0 and i <= self.num_layers //2:
                layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
                layer_inner = self.act_fn(con(layer_inner, adj, h0, self.lamda, self.alpha, i + 1))
                h0 = layer_inner
            else:
                layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
                layer_inner = self.act_fn(con(layer_inner, adj, layer0, self.lamda, self.alpha, i + 1))
        out = F.dropout(layer_inner, self.dropout, training=self.training)
        out = self.layer2(out,adj)
        return out
