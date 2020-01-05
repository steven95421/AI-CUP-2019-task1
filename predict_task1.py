#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
from transformers import AdamW, WarmupLinearSchedule

import nltk
from nltk.corpus import stopwords

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import csv
import random
import matplotlib.pyplot as plt
import sys
import os
import pickle


# In[2]:


use_cuda = torch.cuda.is_available()
if use_cuda:
    print("using cuda!")
print("seed:", sys.argv[1])

if not os.path.exists("model/model_{}_state".format(sys.argv[1])):
    print("NOT FOUND")
    sys.exit()

device = torch.device("cuda:"+sys.argv[2])
col_names = []
x = []
y = []
lens = []
MAX_LEN = 512
rev = False
dct = {'BACKGROUND': 0, 'OBJECTIVES': 1, 'METHODS': 2, "RESULTS": 3, 'CONCLUSIONS': 4, 'OTHERS': 5}
pool = [0 for _ in range(29)] + [1 for _ in range(20)] + [2 for _ in range(29)] +        [3 for _ in range(25)] + [4 for _ in range(11)] + [5 for _ in range(2)]
stop_words = set(stopwords.words('english'))
rm_sw = False


# In[3]:


# ## Fine-Tune Bert

# In[4]:


import math

class gelu(nn.Module):
    
    def __init__(self):
        super(gelu, self).__init__()

    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * cdf

    
class MutiLabelModel(nn.Module):
    
    def __init__(self, encoder, emb_size=768, out_size=6):
        super(MutiLabelModel, self).__init__()
        
        self.encoder = encoder
        self.fn_size = emb_size
        self.out_size = out_size
        
        self.out_fn = nn.Linear(emb_size, out_size)

    def forward(self, inp, seg_inp):

        # batch = number of sentence
        embs = self.encoder(inp, token_type_ids=seg_inp)[0] # [batch, seq, emb_dim]
        outputs = embs[0, (inp==102).nonzero()[:, 1], :] # [batch, emb_dim]
        outputs = self.out_fn(outputs).view(-1, self.out_size) # [batch, out_dim]

        return outputs


# ## Sentences to Tokens

# In[6]:


seed = int(sys.argv[1]) # tune this only
random.seed(seed)
torch.manual_seed(seed)
best_f1 = -1
batch_size = 8
epochs = 3
lr = 1e-5
softmax = False

# load pre-trained bert
print("loading bert...")
tokenizer = BertTokenizer.from_pretrained('scibert_scivocab_uncased')
encoder = BertModel.from_pretrained('scibert_scivocab_uncased')

# ## Train

# In[7]:


# %%time

take = 6

# init model
print("initializing model...")
model = MutiLabelModel(encoder, 768, take)
model.load_state_dict(torch.load("model/model_{}_state".format(seed), map_location=device))

test_data = []
lens = []

with open("data/task1_private_testset.csv") as f:
    rows = csv.reader(f)
    for row in rows:
        if row[0] == "Id":
            col_names = row
            continue
        if rev:
            row[2] = " ".join(row[2].replace('$$$', ' $$$ ').split()[::-1]).replace(' $$$ ', '$$$')
        lens.append(len(row[2].split('$$$')))
        row[2] = row[2].replace('.', '').replace('$$$', ' [CLS] ')
        row[2] = '[CLS] ' + row[2] + ' {} [SEP]'.format(lens[-1])
        test_data.append(row[2])
        
test_data[0]


# In[24]:


count_guess = 0
# sentences to tokens
print("encoding sentences to tokens...")
for i in tqdm(range(len(test_data))):
    indexed_tokens = tokenizer.encode(test_data[i])[:MAX_LEN]
    segments_ids = [0 for _ in range(len(indexed_tokens))]
    if len(indexed_tokens) < MAX_LEN:
        test_data[i] = (torch.LongTensor([indexed_tokens]), torch.LongTensor([segments_ids]))
    else:
        test_data[i] = [lens[i]]
        count_guess += 1
count_guess


# In[25]:
model = model.to(device)
model = model.eval()

thd = 1/take if softmax else 0.7
thd2 = 0.6
thrld = np.ones((1,take))*thd
thrld_ten = torch.from_numpy(thrld).float().to(device)
preds = []

with torch.no_grad():
    pbar = tqdm(enumerate(test_data), total=len(test_data))
    for sidx, tokens in pbar:
        if len(tokens) == 1:
            for i in range(tokens[0]):
                start = int((5/tokens[0]) * i)
                if start >= 4:
                    start = 3
                guess = [0.0 for _ in range(6)]
                guess[start], guess[start+1] = 1.0, 1.0
                preds.append(guess)
        else:
            inp, seg_inp = tokens[0].to(device), tokens[1].to(device)
            out = model(inp, seg_inp)

            if softmax:
                out = torch.softmax(out, 1)
                out = torch.log(out) - torch.log(torch.ones(out.size()).to(device)-out)

            out = torch.sigmoid(out)
            preds += out.tolist()


# In[26]:


for i in range(len(preds)):
    for j in range(len(preds[i])):
        preds[i][j] = round(preds[i][j], 5)
            
preds[0]


# In[27]:


zeros_all = []
with open("data/task1_sample_submission.csv") as f:
    rows = csv.reader(f)
    for row in rows:
        zeros_all.append(row)


# In[28]:


with open("results/task1/result_{}.csv".format(seed), "w", newline="") as f:
    w = csv.writer(f)
    for i, row in enumerate(zeros_all):
        if i >= 131167:
            w.writerow(row[:1]+preds[i-131167])
        else:
            w.writerow(row)