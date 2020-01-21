import torch
from torch import nn
from torch.nn import Module
from models.layers import *
import pandas as pd


class DHGNN_v1(nn.Module):
    """
    Dynamic Hypergraph Convolution Neural Network with a GCN-style input layer
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(
            dim_in=self.dims_in[0],
            dim_out=self.dims_out[0],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[0],
            has_bias=kwargs['has_bias'])]
            + [DHGLayer(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            structured_neighbor=kwargs['k_structured'],
            nearest_neighbor=kwargs['k_nearest'],
            cluster_neighbor=kwargs['k_cluster'],
            wu_knn=kwargs['wu_knn'],
            wu_kmeans=kwargs['wu_kmeans'],
            wu_struct=kwargs['wu_struct'],
            n_cluster=kwargs['clusters'],
            n_center=kwargs['adjacent_centers'],
            has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :param G:
        :return:
        """
        ids = kwargs['ids']
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']
        G = kwargs['G']
        ite = kwargs['ite']

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](ids, x, edge_dict, G, ite)
        return x


class DHGNN_v2(nn.Module):
    """
    Dynamic Hypergraph Convolution Neural Network with a HGNN-style input layer
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([HGNN_conv(
            dim_in=self.dims_in[0],
            dim_out=self.dims_out[0],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[0],
            has_bias=kwargs['has_bias'])]
            + [DHGLayer(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            structured_neighbor=kwargs['k_structured'],
            nearest_neighbor=kwargs['k_nearest'],
            cluster_neighbor=kwargs['k_cluster'],
            wu_knn=kwargs['wu_knn'],
            wu_kmeans=kwargs['wu_kmeans'],
            wu_struct=kwargs['wu_struct'],
            n_cluster=kwargs['clusters'],
            n_center=kwargs['adjacent_centers'],
            has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :param G:
        :return:
        """
        ids = kwargs['ids']
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']
        G = kwargs['G']
        ite = kwargs['ite']

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](ids, x, edge_dict, G, ite)
        return x


