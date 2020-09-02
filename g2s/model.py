from gatenc import GAT
from gptdec import GPT2
from onmt.Models import GCNEncoder_DGL


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


def make_model(opt, src, tgt):
    model = Graph2Seq(
        # GAT(g, in_dim, hidden_dim, out_dim, num_heads),
        GCNEncoder_DGL(src, num_inputs=256, num_units=256,
                 num_labels=768),
        GPT2()
    )

    return model

# gcn_num_inputs=256, gcn_num_labels=5, gcn_num_layers=2, gcn_num_units=256, gcn_out_arcs=True, gcn_residual='residual', gcn_use_gates=False, gcn_use_glus=False

