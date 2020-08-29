# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import numpy as np
# from torchtext import data 
# import torchtext
from pathlib import Path
import pandas as pd
# import spacy

# # You'll probably need to use the 'python' engine to load the CSV
# # tweetsDF = pd.read_csv("training.1600000.processed.noemoticon.csv", header=None)
# tweetsDF = pd.read_csv("training.1600000.processed.noemoticon.csv", 
# engine="python", header=None)

# print(tweetsDF[0].value_counts())

# tweetsDF["sentiment_cat"] = tweetsDF[0].astype('category')
# tweetsDF["sentiment"] = tweetsDF["sentiment_cat"].cat.codes
# tweetsDF.to_csv("train-processed.csv", header=None, index=None)      
# # tweetsDF.sample(10000).to_csv("train-processed-sample.csv", header=None, index=None)

df = pd.read_csv("train-processed.csv")
df.sample(100).to_csv("train-processed-100samples.csv", index=None, header=None)