# Using PyTorch 1.4

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import random
import torch
import fire
import logging
import os
import csv

from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F

from gpt2 import GPT2
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm

class CSVTwitter(Dataset):
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=768):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.tweets = []

        # This uses the same CSV of Sentiment140 that we created in Chapter 5
        
        with open('train-processed-100samples.csv', newline='') as csvfile:
            tweet_csv = csv.reader(csvfile)
            for row in tweet_csv:
                self.tweets.append(torch.tensor(
                    self.tokenizer.encode(f"<|{control_code}|>{row[5][:max_length]}<|endoftext|>")
                ))
                
        if truncate:
            self.tweets = self.tweets[:20000]
        self.tweet_count = len(self.tweets)
        
    def __len__(self):
        return self.tweet_count

    def __getitem__(self, item):
        return self.tweets[item]

def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

def train(
    dataset,
    model,
    tokenizer,
    batch_size=16,
    epochs=4,
    lr=2e-5,
    max_seq_len=400,
    warmup_steps=5000,
    gpt2_type="gpt2",
    device="cuda",
    output_dir=".",
    output_prefix="wreckgar",
    test_mode=False,
    save_model_on_epoch=False,
):

    acc_steps = 100

    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            # outputs = model(input_tensor)
            print("input_tensor.size", input_tensor.size())
            # outputs = model(input_tensor)
            outputs = model(input_tensor, labels=input_tensor)
            print("outputs.size", len(outputs))
            # loss = outputs
            loss = outputs[0]
            print("loss.size", loss.size())
            print("loss[0]", loss.item())
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model

print("\nloading dataset\n")

dataset = CSVTwitter("<|tweet|>", truncate=True, gpt2_type="gpt2")
gpt2_type = "gpt2"

print("\ndataset loaded\n")

model_g = GPT2()

model_dict = model_g.state_dict() #currently with random initialization
state_dict = torch.load("./gpt2-pytorch_model.bin") #pretrained weights

old_keys = []
new_keys = []
for key in state_dict.keys(): 
    if "mlp" in key: #The hugging face state dict references the feedforward network as mlp, need to replace to `feedforward` be able to reuse these weights
        new_key = key.replace("mlp", "feedforward")
        new_keys.append(new_key)
        old_keys.append(key)

for old_key, new_key in zip(old_keys, new_keys): 
    state_dict[new_key]=state_dict.pop(old_key)

pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}


model_dict.update(pretrained_dict)
model_g.load_state_dict(model_dict)

model_g_tr = train(
    dataset,
    model_g,
    # GPT2LMHeadModel.from_pretrained(gpt2_type),
    GPT2Tokenizer.from_pretrained(gpt2_type),
    batch_size=16,
    epochs=1,
    lr=3e-5,
    max_seq_len=140,
    warmup_steps=5000,
    gpt2_type=gpt2_type,
    device="cuda",
    output_dir="trained_models",
    output_prefix="twitter",
    save_model_on_epoch=True
)

tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
context   = torch.tensor([tokenizer.encode("<|tweet|> The planet earth")])
print(context)

def generate(context, ntok=20):
    for _ in range(ntok):
        out = model(context)
        logits = out[:, -1, :]
        indices_to_remove = logits < torch.topk(logits, 10)[0][..., -1, None]
        logits[indices_to_remove] = np.NINF
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
        context = torch.cat([context, next_tok.unsqueeze(-1)], dim=-1)
    return context

out = generate(context, ntok=20)
print(tokenizer.decode(out[0]))


# def generate(
#     model,
#     tokenizer,
#     prompt,
#     entry_count=10,
#     entry_length=100,
#     top_p=0.8,
#     temperature=1.,
# ):

#     model.eval()

#     generated_num = 0
#     generated_list = []

#     filter_value = -float("Inf")

#     with torch.no_grad():

#         for entry_idx in trange(entry_count):

#             entry_finished = False

#             generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
#             generated = generated.to("cuda") # get the dimension of generated

#             # Using top-p (nucleus sampling): https://github.com/huggingface/transformers/blob/master/examples/run_generation.py

#             for i in range(entry_length):
#                 outputs = model(generated)
                
#                 # outputs = model(generated, labels=generated)
#                 # loss, logits = outputs[:2]

#                 logits = outputs
                
#                 logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

#                 sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#                 cumulative_probs = torch.cumsum(
#                     F.softmax(sorted_logits, dim=-1), dim=-1
#                 )

#                 sorted_indices_to_remove = cumulative_probs > top_p
#                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
#                     ..., :-1
#                 ].clone()
#                 sorted_indices_to_remove[..., 0] = 0

#                 indices_to_remove = sorted_indices[sorted_indices_to_remove]
#                 logits[:, indices_to_remove] = filter_value

#                 next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
#                 generated = torch.cat((generated, next_token), dim=1)

#                 if next_token in tokenizer.encode("<|endoftext|>"):
#                     entry_finished = True

#                 if entry_finished:

#                     generated_num = generated_num + 1

#                     output_numpy = generated.squeeze().cpu()
#                     output_list = list(output_numpy)

#                     # output_list = list(generated.squeeze().numpy())
#                     output_text = tokenizer.decode(output_list)

#                     generated_list.append(output_text)
#                     break
            
#             if not entry_finished:
#                 output_numpy = generated.squeeze().cpu()
#                 output_list = list(output_numpy)
#                 # output_list = list(generated.squeeze().numpy())
#                 output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
#                 generated_list.append(output_text)
                
#     return generated_list



# generated_tweets = generate(model_g_tr, GPT2Tokenizer.from_pretrained(gpt2_type),"<|tweet|>",entry_count=10)
# # generated_tweets = generate(model.to('cpu'), GPT2Tokenizer.from_pretrained(gpt2_type),"<|tweet|>",entry_count=10)

# print(generated_tweets)