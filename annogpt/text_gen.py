from transformers import GPT2Tokenizer
from gpt2 import GPT2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm
import os
from tqdm import tqdm_notebook, trange
import logging


# model = GPT2()
# model_dict = model.state_dict() #currently with random initialization
# state_dict = torch.load("./gpt2-pytorch_model.bin") #pretrained weights

# old_keys = []
# new_keys = []
# for key in state_dict.keys(): 
#     if "mlp" in key: #The hugging face state dict references the feedforward network as mlp, need to replace to `feedforward` be able to reuse these weights
#         new_key = key.replace("mlp", "feedforward")
#         new_keys.append(new_key)
#         old_keys.append(key)

# for old_key, new_key in zip(old_keys, new_keys): 
#     state_dict[new_key]=state_dict.pop(old_key)

# pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}

# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
# model.eval() #model in inference mode as it's now initialized with pretrained weights

# print(model)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
context   = torch.tensor([tokenizer.encode("The planet earth")])
print(context)

# def generate(context, ntok=20):
#     for _ in range(ntok):
#         out = model(context)
#         logits = out[:, -1, :]
#         indices_to_remove = logits < torch.topk(logits, 10)[0][..., -1, None]
#         logits[indices_to_remove] = np.NINF
#         next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
#         context = torch.cat([context, next_tok.unsqueeze(-1)], dim=-1)
#     return context

# out = generate(context, ntok=20)
# print(tokenizer.decode(out[0]))

