import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam
from torch.profiler import profile, record_function, ProfilerActivity
import networkx as nx
from scipy.sparse import csr_matrix
from GCN import *
import os
import random
import numpy as np
import math
from collections import Counter
from config import *
from tqdm import tqdm,trange

def csr2dense(edge):
    row=np.array(edge[0])
    col=np.array(edge[1])
    data=np.array(torch.ones_like(edge[0]))
    coo_matrix=csr_matrix((data,(row,col)),shape=[node_max,node_max])
    dense=torch.tensor(coo_matrix.todense(),dtype=torch.float32)
    return dense

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(2, 3)
        score = q @ k_t
        score = score / math.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        v = score @ v
        return v, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head,device):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model,device=device)
        self.w_k = nn.Linear(d_model, d_model,device=device)
        self.w_v = nn.Linear(d_model, d_model,device=device)
        self.w_concat = nn.Linear(d_model, d_model,device=device)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)
        mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out) # [batch_size, length, d_model]
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, seq_len):

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob,device):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head,device=device)
        self.norm1 = LayerNorm(d_model=d_model).to(device)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model).to(device)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob).to(device)

    def forward(self, dec, mask=None):
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=mask)
        # x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        # x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Decoder(nn.Module):
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob,
                                                  device=device)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, d_model,device=device)

    def forward(self, emb, mask):
        for layer in self.layers:
            trg = layer(emb,mask)

        # pass to LM head
        output = self.linear(trg)
        return output

class DGPretrain(torch.nn.Module):
    def __init__(self,input_dim,hid_dim, ffn_hidden, n_head, layer_num, drop_prob,device):
        super().__init__()
        self.hid_dim=hid_dim
        self.device = device
        self.gnn = GCNConv(input_dim,self.hid_dim).to(self.device)
        self.readout = nn.Linear(node_max, 1, bias=False)

        self.pos_emb = PositionalEncoding(self.hid_dim, seq_len,self.device).to(self.device)
        
        self.decoder = Decoder(seq_len, self.hid_dim, ffn_hidden, n_head, layer_num, drop_prob, self.device) # 解码器，用于学习文本生成能力
        self.linear = nn.Sequential(
            nn.Linear(self.hid_dim+self.hid_dim,self.hid_dim),
            nn.PReLU()
        )

    def make_batch(self,egonet_edges,egonet_x,model_type):
        print('make batch',model_type)
        ego_batch = []
        if model_type=='train':
            selected_indices = random.sample(range(len(egonet_x)), batch_size)
        elif model_type=='link_prd':
            selected_indices=range(len(egonet_x))
        for index in tqdm(selected_indices):
            seq=[]
            node_seq=[]
            # print(egonet_edges[index][6])
            for time in range(len(egonet_x[index])):
                node_emb=self.gnn(egonet_x[index][time].to(self.device),egonet_edges[index][time].to(self.device))
                node_seq.append(node_emb)
                # ego_emb=self.readout(node_emb.T).squeeze(1)
                # ego_emb=node_emb.mean(dim=0, keepdim=False)
                # ego_emb, indices=torch.max(node_emb, dim=0)
                # print(ego_emb.shape)
                seq.append(node_emb[0])
            # print(node_seq[5][0])
            # print(seq[6])
            postion_emb=self.pos_emb(len(egonet_x[index]))
            for token in range(len(egonet_x[index])):
                seq[token]=seq[token]+postion_emb[token]
            ego_batch.append(torch.stack(seq,dim=0))

        # print(ego_batch[4][6])
        # print(ego_batch[64][6])
        # print(ego_batch[267][6])
        ego_batch=torch.stack(ego_batch,dim=0)
        # node_batch是节点的embedding，大小为[batch_size,seq_len,nodes_num,hidden_dim]  
        # input_batch, output_batch都是graph embedding，shape为[batch_size,seq_len,hidden_dim]
        # 后两个不是embedding，就是节点集和边集，因为mini batch随机选，还真得保存下来
        return ego_batch

    # def make_eval(self,edges,x):
    #     sos=torch.zeros(self.hid_dim,device=self.device)
    #     emb_eval=[]
    #     seq=[sos]
    #     for time in range(len(x)):
    #         graph_emb,node_emb=self.gnn(x[time].to(self.device),edges[time].to(self.device))
    #         seq=seq+[graph_emb]
    #     postion_emb=self.pos_emb(len(x))
    #     for token in range(len(x)):
    #         seq[token]=seq[token]+postion_emb[token]
    #     emb_eval.append(torch.stack(seq,dim=0))
    #     emb_eval=torch.stack(emb_eval,dim=0)
    #     return emb_eval

    # def loss_re(self,logits, target_adj_dense):
    #     temp_size = target_adj_dense.size()[0]
    #     temp_sum = target_adj_dense.sum()
    #     posw = float(temp_size * temp_size - temp_sum) / temp_sum
    #     norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
    #     nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits
    #                                                       , target=target_adj_dense
    #                                                       , pos_weight=posw
    #                                                       , reduction='none')
    #     nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0,1])
    #     return - nll_loss
    
    # output size:[seq_len,nodes_num,dim]


    def get_attn_subsequent_mask(self,seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]  
        # 使用 numpy 创建一个上三角矩阵（triu = triangle upper）
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        #------------------------- 维度信息 --------------------------------
        # subsequent_mask 的维度是 [batch_size, seq_len(Q), seq_len(K)]
        subsequent_mask = torch.from_numpy(subsequent_mask).byte()
        return subsequent_mask

    # model_type用于控制训练以及下游任务，例如链路预测，center_node是节点对，与训练过程不一样
    # egonet_center不一样，在训练过程中是一个点，link_prd是两个点一对
    def forward(self,egonet_edges,egonet_x,model_type):
        # if model_type=='train':
        ego_batch = self.make_batch(egonet_edges,egonet_x,model_type)
        mask=self.get_attn_subsequent_mask(ego_batch).to(self.device)
        prd_batch=self.decoder(ego_batch,mask)
        # elif model_type=='link_prd':
        #     ego_batch0 = self.make_batch(egonet_edges,egonet_x,model_type,0)
        #     mask=self.get_attn_subsequent_mask(ego_batch0).to(self.device)
        #     prd_batch0=self.decoder(ego_batch0,mask)     
        #     ego_batch1 = self.make_batch(egonet_edges,egonet_x,model_type,1)
        #     prd_batch1=self.decoder(ego_batch1,mask)     
        #     prd_batch=[prd_batch0,prd_batch1]
        #     ego_batch=[ego_batch0,ego_batch1]
        return prd_batch,ego_batch