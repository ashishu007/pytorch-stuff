# Standard PyTorch imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
import matplotlib.pyplot as plt

from s2s.attn import MultiHeadedAttention
from s2s.encoder import Encoder, EncoderLayer
from s2s.decoder import Decoder, DecoderLayer
from s2s.utils import PositionalEncoding, PositionwiseFeedForward, Embeddings, Generator
from s2s.gpt import GPT, GPTConfig

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base model for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encoder(self.src_embed(src), src_mask)
        output = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return output


def make_model(src_vocab, tgt_vocab, N=2, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Construct a model object based on hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        # GPT(GPTConfig(tgt_vocab, src_vocab, n_layer=N, n_head=h, n_embd=d_model)),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. Initialize parameters with Glorot or fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


# Small example model.
tmp_model = make_model(10, 10, 1)
print(tmp_model)

# from s2s.gpt import GPT, GPTConfig
# mconf = GPTConfig(10, 10, n_layer=1, n_head=8, n_embd=512) # a GPT-1
# model = GPT(mconf)
# print(model)