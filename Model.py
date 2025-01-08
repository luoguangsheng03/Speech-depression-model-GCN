# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:11:33 2023
The defination of GCN model
@author: 50357
"""

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        #self.gc2 = GraphConvolution(nhid, nclass)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc3 = GraphConvolution(nhid2, nhid3)
        self.gc4 = GraphConvolution(nhid3, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        #x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)
    
    
class GCN_two_layers(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super(GCN_two_layers, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        #self.gc2 = GraphConvolution(nhid, nclass)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc3 = GraphConvolution(nhid2, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        
        #x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)
