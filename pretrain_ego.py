import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam
from torch.profiler import profile, record_function, ProfilerActivity
import networkx as nx
import os
import random
import numpy as np
import math
torch.autograd.set_detect_anomaly(True)
from collections import Counter
from config import *
from model import DGPretrain
from tqdm import tqdm,trange
import pickle
from torch.profiler import profile, record_function, ProfilerActivity
torch.set_printoptions(profile="full")

with open('/home/sdc/gjq/code_repository/Pre/egonet_data/edges_stack_cite.pkl', 'rb') as file:
    egonet_edges=pickle.load(file)
# with open('/home/ltp/code_repository/NewWork/egonet_data/x_stack_sep.pkl', 'rb') as file:
#     egonet_x=pickle.load(file)
# with open('/home/ltp/code_repository/NewWork/egonet_data/center_stack_sep.pkl', 'rb') as file:
#     egonet_center=pickle.load(file)
print('共有egonet{}个'.format(len(egonet_edges)))
# selected_indices = random.sample(range(len(egonet_edges_raw)), 2100)
# egonet_edges=[egonet_edges_raw[i] for i in selected_indices]
egonet_x=[]
for egonet in egonet_edges:
    x=[]
    node_features = torch.tensor(np.eye(node_max).astype(np.float32))
    for t in range(len(egonet)):
        x.append(node_features)
    egonet_x.append(x)

def loss_re(emb_re,target):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    batch_size, time_length, emb_dim = emb_re.shape
    loss=0
    for i in range(batch_size):
        for t in range(time_length):
            loss+=(1-cos(emb_re[i][t], target[i][t]))
    return loss

def reconstruction(prd,ego):
    loss=self.loss_re(prd_batch[:][:-1],ego_batch[:][1:])
    return loss





model=DGPretrain(input_dim=node_max,hid_dim = hid_dim, ffn_hidden=ffn_hidden, n_head=n_head, layer_num = layer_num, drop_prob=drop_prob,device = device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     loss,graph_re,emb_re=model(egonet_edges,egonet_x)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
patient=0
loss_earlystop=100000
print('模型训练')
# print(egonet_edges[0][4])
# print(egonet_edges[64][4])
for i in range(epochs):
    optimizer.zero_grad()
    prd_batch,ego_batch=model(egonet_edges,egonet_x,'train')
    # print(prd_batch[0][4])
    # print(prd_batch[64][4])
    # print(ego_batch[0][5])
    # print(ego_batch[64][5])
    # print('loss backward')
    loss=loss_re(prd_batch[:][:-1],ego_batch[:][1:])
    loss.backward()
    optimizer.step()
    print("----------------------------------")
    print('epoch: ', i)
    print(f"loss ={loss:.8f}")
    if loss<loss_earlystop:
        loss_earlystop=loss
        patient=0
    else:
        patient+=1
    if patient==10:
        break
torch.save(model.state_dict(), 'pretrain_dict_cite.pt')
torch.save(model, 'pretrain_full_cite.pt')