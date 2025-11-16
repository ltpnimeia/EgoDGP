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

def contrastive_next_step_loss(predicted_embs, target_embs, temperature=0.07):
    """
    使用batch内负样本的对比学习损失（InfoNCE Loss）

    Args:
        predicted_embs: [B, L-1, d] - batch内所有序列的预测
        target_embs: [B, L-1, d] - batch内所有序列的目标
        temperature: 温度参数，默认0.07

    Returns:
        InfoNCE对比学习损失

    损失函数形式:
        L_gen = -1/(B(L-1)) Σ_{i=1}^{B(L-1)} log [exp(⟨ê_i, e_i⟩/τ) / Σ_{j=1}^{B(L-1)} exp(⟨ê_i, e_j⟩/τ)]

    其中:
        - B是batch size
        - i索引所有batch内的时间步
        - ê_i是预测的embedding
        - e_i是对应的真实embedding
        - 分母中的负样本来自batch内其他时间步和序列
    """
    B, L_minus_1, d = predicted_embs.shape

    # 展平为 [B*(L-1), d]
    pred_flat = predicted_embs.reshape(-1, d)
    target_flat = target_embs.reshape(-1, d)

    # 归一化
    pred_flat = F.normalize(pred_flat, dim=-1)
    target_flat = F.normalize(target_flat, dim=-1)

    # 计算相似度矩阵 [B*(L-1), B*(L-1)]
    sim_matrix = torch.matmul(pred_flat, target_flat.T) / temperature

    # 对角线是正样本(每个预测对应其真实目标)
    labels = torch.arange(B * L_minus_1).to(pred_flat.device)

    # InfoNCE损失
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


# 保留原有的损失函数作为备选（已弃用）
def loss_re(emb_re,target):
    """已弃用：原始的余弦相似度损失函数"""
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
    # 使用对比学习损失（InfoNCE）替代原有的余弦相似度损失
    loss=contrastive_next_step_loss(prd_batch[:][:-1],ego_batch[:][1:])
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