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
import pickle
import os

# In[2]:

if os.path.exists("model/model_{}_state".format(sys.argv[1])):
    print("MODEL TRAINED!")
    sys.exit()

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("using cuda!")
print("seed:", sys.argv[1])
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

with open("data/task1_trainset.csv") as f:
    rows = csv.reader(f)
    for row in rows:
        if row[0] == "Id":
            col_names = row
            continue
            
        if rev:
            row[2] = " ".join(row[2].replace('$$$', ' $$$ ').split()[::-1]).replace(' $$$ ', '$$$')
            
        if rm_sw:
            row[2] = " ".join([word for word in row[2].split() if word.lower() not in stop_words])
            
        row[2] = row[2].replace('.', '').replace('$$$', ' [CLS] ')
        row[2] = "[CLS] " + row[2]
        row[6] = row[6].split(' ')
        row[2] = row[2] + ' {} [SEP]'.format(len(row[6]))
        
        tmp_y = []
        
        for i, v in enumerate(row[6]):
            label = [0 for _ in range(6)]
            l = v.split('/')
            for c in l:
                label[dct[c]] = 1
            tmp_y.append(label)
           
        x.append(row[2])
        y.append(tmp_y)
        lens.append(len(row[2].split()))
            
x = np.array(x)
y = np.array(y)
assert len(y) == len(x)
x[0], y[0], len(x)


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


# In[5]:


def micro_f1_score(pred, label):
    '''
    parameters:
        pred, label: [batch, out_dim]
    return:
        f1: float
        TP, FP, FN: [out_dim, ]
    '''
    TP = torch.sum(pred*label, 0)
    FP = torch.sum(pred*(label-1) != 0, 0)
    FN = torch.sum((pred-1)*label != 0, 0)
    
    precision = TP.sum().item() / (TP.sum().item() + FP.sum().item() + 1e-10)
    recall = TP.sum().item() / (TP.sum().item() + FN.sum().item() + 1e-10)
    
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return f1, TP, FP, FN


# ## Sentences to Tokens

# In[6]:


Train_x = []
Train_y = []
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

# sentences to tokens
print("encoding sentences to tokens...")
for i in tqdm(range(len(x))):
    indexed_tokens = tokenizer.encode(x[i])[:MAX_LEN]
    segments_ids = [0 for _ in range(len(indexed_tokens))]
    if len(indexed_tokens) < MAX_LEN:
        Train_x.append((torch.LongTensor([indexed_tokens]), torch.LongTensor([segments_ids])))
        Train_y.append(y[i])
        
train_idxs = random.sample([i for i in range(len(Train_x))], round(len(Train_x)*0.9))
test_idxs = [i for i in range(len(Train_x)) if i not in train_idxs]
train_x, train_y = [Train_x[idx] for idx in train_idxs], [Train_y[idx] for idx in train_idxs]
test_x, test_y = [Train_x[idx] for idx in test_idxs], [Train_y[idx] for idx in test_idxs]

print("train len: {}\ttest len: {}".format(len(train_x), len(test_x)))
print("train ratios:", end=" ")
for i in range(6):
    print(sum([sum([y[i] for y in ys]) for ys in train_y])/sum([len(ys) for ys in train_y]), end="\t")
print("")
print("dev ratios:", end="   ")
for i in range(6):
    print(sum([sum([y[i] for y in ys]) for ys in test_y])/sum([len(ys) for ys in test_y]), end="\t")
print("")

# ## Train

# In[7]:


# %%time

take = 6

# init model
print("initializing model...")
model = MutiLabelModel(encoder, 768, take)
pos_weight = torch.FloatTensor([2., 2.25, 2., 2.87, 4., 8.7]).to(device)
num_total_steps = np.ceil(len(train_x) / batch_size)*epochs
num_warmup_steps = int(num_total_steps * 0.5)

optim = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = WarmupLinearSchedule(optim, warmup_steps=num_warmup_steps, t_total=num_total_steps)

model = model.to(device)

# model.load_state_dict(torch.load("/tmp2/r08922010/aicup/model/task1/model_{}_state".format(seed), map_location=device))
# torch.save(model.encoder.state_dict(), "/tmp2/r08922010/aicup/model/task1/encoder_{}_state".format(seed))

thd = 1/take if softmax else 0.7
thd2 = 0.6
thrld = np.ones((1,take))*thd

outs = [[] for _ in range(take)]
dev_micro_f1 = -1

# train and validation

print("start training!")
for ep in range(epochs):
    model = model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss = 0.0

    train_metric = { "TP":np.zeros(take), "FP":np.zeros(take), "FN":np.zeros(take), "F1":[] }
    if use_cuda:
        thrld_ten = torch.from_numpy(thrld).float().to(device)
        train_metric["TP"] = torch.from_numpy(train_metric["TP"]).float().to(device)
        train_metric["FP"] = torch.from_numpy(train_metric["FP"]).float().to(device)
        train_metric["FN"] = torch.from_numpy(train_metric["FN"]).float().to(device)
    else:
        thrld_ten = torch.from_numpy(thrld).float()

    optim.zero_grad()
    pbar = tqdm(enumerate(train_x), total=len(train_x))
    
    for sidx, tokens in pbar:

        if (sidx+1) % batch_size == 0 and sidx:
            optim.step()
            scheduler.step()
            optim.zero_grad()

        # abstract
        inp, seg_inp = tokens[0].to(device), tokens[1].to(device)
        target = torch.FloatTensor(train_y[sidx]).to(device)

        out = model(inp, seg_inp)

        l = criterion(out, target)
        total_loss += l.item() / target.size(0)

        l.backward()

        out = torch.sigmoid(out)
        pred1 = (out > thrld_ten.expand(target.size())).float()
        max_idx = torch.argmax(out, 1, keepdim=True)
        one_hot = torch.zeros(out.size()).to(device)
        pred2 = one_hot.scatter_(1, max_idx, 1)
        pred = (pred1+pred2 >= 1).float()
        
        # post process
        pred = pred.tolist()
        for i in range(len(pred)):
            tmp = sum(pred[i])
            if tmp == 0.0:
                pred[i][int((5/len(pred))*i)] = 1.0
            elif tmp == 1.0:
                idx = pred[i].index(1.0)
                if idx == 0:
                    if out[i][1] > thd2:
                        pred[i][1] = 1.0
                elif 1 <= idx <= 3:
                    choose = idx-1 if out[i][idx-1] >= out[i][idx+1] else idx+1
                    if out[i][choose] > thd2:
                        pred[i][choose] = 1.0
                elif idx == 4:
                    if out[i][3] > thd2:
                        pred[i][3] = 1.0
        pred = torch.FloatTensor(pred).to(device)
            
        f1, tp, fp, fn = micro_f1_score(pred, target)

        train_metric["F1"].append(f1)
        train_metric["TP"] += tp.float()
        train_metric["FP"] += fp.float()
        train_metric["FN"] += fn.float()

    optim.step()
    scheduler.step()

    train_precision_all = train_metric["TP"].sum().item() / (train_metric["TP"].sum().item() + train_metric["FP"].sum().item())
    train_recall_all = train_metric["TP"].sum().item() / (train_metric["TP"].sum().item() + train_metric["FN"].sum().item())
    train_micro_f1 = (2 * train_precision_all * train_recall_all) / (train_precision_all + train_recall_all)

    avg_loss = total_loss / len(train_x)

    print("Train Loss:{}\tmicro_f1:{}".format(avg_loss, train_micro_f1))
    print("micro_f1s:", end="")
    for i in range(take):
        precision = train_metric["TP"][i].item() / (train_metric["TP"][i].item()+train_metric["FP"][i].item()+1e-10)
        recall = train_metric["TP"][i].item() / (train_metric["TP"][i].item()+train_metric["FN"][i].item()+1e-10)
        print("{}".format(2*precision*recall / (precision+recall+1e-10)), end="\t")
    print("")

    # Evaluation
    model = model.eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dev_loss = 0.0
    dev_metric = { "TP":np.zeros(take), "FP":np.zeros(take), "FN":np.zeros(take), "F1":[] }
    if use_cuda:
        dev_metric["TP"] = torch.from_numpy(dev_metric["TP"]).float().to(device)
        dev_metric["FP"] = torch.from_numpy(dev_metric["FP"]).float().to(device)
        dev_metric["FN"] = torch.from_numpy(dev_metric["FN"]).float().to(device)

    with torch.no_grad():

        for sidx, tokens in tqdm(enumerate(test_x), total=len(test_x)):
            
            inp, seg_inp = tokens[0].to(device), tokens[1].to(device)
            target = torch.FloatTensor(test_y[sidx]).to(device)

            out = model(inp, seg_inp)
        
            if softmax:
                out = torch.softmax(out, 1)
                out = torch.log(out) - torch.log(torch.ones(out.size()).to(device)-out)

            l = criterion(out, target)
            dev_loss += l / target.size(0)

            out = torch.sigmoid(out)
            pred1 = (out > thrld_ten.expand(target.size())).float()
            max_idx = torch.argmax(out, 1, keepdim=True)
            one_hot = torch.zeros(out.size()).to(device)
            pred2 = one_hot.scatter_(1, max_idx, 1)
            pred = (pred1+pred2 >= 1).float()
            
            # post process
            pred = pred.tolist()
            for i in range(len(pred)):
                tmp = sum(pred[i])
                if tmp == 0.0:
                    pred[i][int((5/len(pred))*i)] = 1.0
                elif tmp == 1.0:
                    idx = pred[i].index(1.0)
                    if idx == 0:
                        if out[i][1] > thd2:
                            pred[i][1] = 1.0
                    elif 1 <= idx <= 3:
                        choose = idx-1 if out[i][idx-1] >= out[i][idx+1] else idx+1
                        if out[i][choose] > thd2:
                            pred[i][choose] = 1.0
                    elif idx == 4:
                        if out[i][3] > thd2:
                            pred[i][3] = 1.0
            pred = torch.FloatTensor(pred).to(device)
            
            f1, tp, fp, fn = micro_f1_score(pred, target)
            if ep == epochs-1:
                for i in range(take):
                    outs[i] += out[:, i].view(-1).tolist()

            dev_metric["F1"].append(f1)
            dev_metric["TP"] += tp.float()
            dev_metric["FP"] += fp.float()
            dev_metric["FN"] += fn.float()

            if use_cuda:
                target = target.cpu()

        dev_precision_all = dev_metric["TP"].sum().item() / (dev_metric["TP"].sum().item() + dev_metric["FP"].sum().item())
        dev_recall_all = dev_metric["TP"].sum().item() / (dev_metric["TP"].sum().item() + dev_metric["FN"].sum().item())
        dev_micro_f1 = (2 * dev_precision_all * dev_recall_all) / (dev_precision_all + dev_recall_all)

        avg_loss = dev_loss / len(test_x)

        print("Dev Loss:{}\tmicro_f1:{}".format(avg_loss.item(), dev_micro_f1))
        print("micro_f1s:", end="")
        for i in range(take):
            precision = dev_metric["TP"][i].item() / (dev_metric["TP"][i].item()+dev_metric["FP"][i].item()+1e-10)
            recall = dev_metric["TP"][i].item() / (dev_metric["TP"][i].item()+dev_metric["FN"][i].item()+1e-10)
            print("{}".format(2*precision*recall / (precision+recall+1e-10)), end="\t")
        print("")

        if dev_micro_f1 > best_f1:
            best_f1 = dev_micro_f1

if dev_micro_f1 > 0.735:
    try:
        with open("seed2f1_task1.pkl", "rb") as f:
            seed2f1 = pickle.load(f)
        with open("seed2f1_task1.pkl", "wb") as f:
            seed2f1[seed] = dev_micro_f1
            pickle.dump(seed2f1, f)
    except:
        print("seed2f1_task1.pkl NOT FOUND")
        with open("seed2f1_task1.pkl", "wb") as f:
            seed2f1 = {}
            seed2f1[seed] = dev_micro_f1
            pickle.dump(seed2f1, f)
    torch.save(model.state_dict(), "model/model_{}_state".format(seed))