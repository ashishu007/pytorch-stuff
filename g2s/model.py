from gatenc import GAT
from gptdec import GPT2

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Graph2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Graph2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.src = src
        # self.tgt = tgt

    def forward(self, features):
        graph_enc = self.encoder(features)
        # avg_enc = torch.mean(graph_enc, 1)
        output = self.decoder(graph_enc)
        return output

def make_model(g, in_dim, hidden_dim, out_dim, num_heads):
    model = Graph2Seq(
        GAT(g, in_dim, hidden_dim, out_dim, num_heads),
        GPT2()
    )

    return model


from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx

def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, mask

import time
import numpy as np

g, features, labels, mask = load_cora_data()


tmp_model = make_model(g, 512, 1024, 768, 2)
# print(tmp_model)

# create optimizer
optimizer = torch.optim.Adam(tmp_model.parameters(), lr=1e-3)

print("type(g)", type(g))
print("features.size()", features.size())
print("len(labels)", len(labels))
print("labels[0]", labels[0])

# main loop
dur = []
for epoch in range(1):
    if epoch >= 3:
        t0 = time.time()

    logits = tmp_model(features)
    print("logits.size()", logits.size())
    print("logits[0]", logits[0])
    
    logp = F.log_softmax(logits, 1)
    # print(logp.size())
    # print(logp[0])
    
    loss = F.nll_loss(logp[mask], labels[mask])

    # t1 = time.time()
    # print(t1 - t0)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), np.mean(dur)))
