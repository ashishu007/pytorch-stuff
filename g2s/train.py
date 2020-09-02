# from gatenc import GAT
# from gptdec import GPT2
# from onmt.Models import GCNEncoder_DGL
from onmt.ModelConstructor import make_embeddings
from utils import lazily_load_dataset, load_fields, tally_parameters
from model import make_model

import torch
import torch.nn as nn
import torch.nn.functional as F

import glob, argparse, opts
import numpy as np

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

opts.gcn_opts(parser)  # adds GCN options

model_opt = parser.parse_args()

dataset = next(lazily_load_dataset("train"))
print(dataset.examples[0].__dict__)
print(dataset)

data_type = dataset.data_type     # data_type: GCN
# Load fields generated from preprocess phase.
fields = load_fields(dataset, data_type, None) # checkpoint = None
print(type(fields))
print(fields)


src_dict = fields["src"].vocab
feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
src_embeddings = make_embeddings(model_opt, src_dict,
                                    feature_dicts)

tgt_dict = fields["tgt"].vocab
feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                    feature_dicts, for_encoder=False)

make_model(model_opt, src_embeddings, tgt_embeddings)

# gcn_num_inputs=256, gcn_num_labels=5, gcn_num_layers=2, gcn_num_units=256, gcn_out_arcs=True, gcn_residual='residual', gcn_use_gates=False, gcn_use_glus=False

# python3 train.py -data data/${model_id}_exp -save_model data/${save_model_name} -encoder_type ${encoder} -encoder2_type ${encoder2} -layers 1 -gcn_num_layers 2 -gcn_num_labels 5 -gcn_residual residual -word_vec_size ${emb_size} -rnn_size ${hidden_size} -gcn_num_inputs ${hidden_size} -gcn_num_units ${hidden_size} -epochs 20 -optim adam -learning_rate 0.001 -learning_rate_decay 0.7 -seed 1 -gpuid 0 -start_checkpoint_at 15 -gcn_in_arcs -gcn_out_arcs -copy_attn -brnn -use_dgl


# model = make_model()  # D: gcn features must be passed here
# tally_parameters(model)     # print the parameter size
# check_save_model_path()     # check if the model path exist
