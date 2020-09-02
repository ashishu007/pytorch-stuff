from gatenc import GAT
from gptdec import GPT2
from onmt.Models import GCNEncoder_DGL
from utils import lazily_load_dataset, load_fields, tally_parameters

import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import numpy as np

class Graph2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Graph2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.src = src
        # self.tgt = tgt

    def forward(self, features):
        h1, h2, memory_bank = self.encoder(features)
        # graph_enc = self.encoder(features)
        # avg_enc = torch.mean(graph_enc, 1)
        output = self.decoder(memory_bank)
        return output

def make_model(g, in_dim, hidden_dim, out_dim, num_heads):
    model = Graph2Seq(
        # GAT(g, in_dim, hidden_dim, out_dim, num_heads),
        GCNEncoder_DGL(),
        GPT2()
    )

    return model

dataset = next(lazily_load_dataset("train"))
print(dataset.examples[0].__dict__)
print(dataset)

data_type = dataset.data_type     # data_type: GCN
# Load fields generated from preprocess phase.
fields = load_fields(dataset, data_type, None) # checkpoint = None
print(type(fields))
print(fields)

# gcn_num_inputs=256, gcn_num_labels=5, gcn_num_layers=2, gcn_num_units=256, gcn_out_arcs=True, gcn_residual='residual', gcn_use_gates=False, gcn_use_glus=False

# model = make_model()  # D: gcn features must be passed here
# tally_parameters(model)     # print the parameter size
# check_save_model_path()     # check if the model path exist
