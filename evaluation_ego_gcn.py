import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
# from torch_geometric import nn as tgnn
import scipy.sparse as sp
from scipy.linalg import block_diag
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import tarfile
import torch.nn.functional as F
import copy
import time
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.datasets import Planetoid
import networkx as nx
import scipy.io as sio
from scipy.sparse import csr_matrix
import torch_scatter
import inspect
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import copy
import pickle
import os
from sklearn.cluster import KMeans
from GCN import GCN

from model import DGPretrain
from config import *

# threshold = 663
# seq_len=9
# seq_start=0
# seq_end = seq_len - 5
# folder_path='/home/gjq/code_repository/NewWork/data/fb_test/fb'

threshold = 6000
seq_len=10
seq_start=0
seq_end = seq_len - 4
folder_path='/home/gjq/code_repository/NewWork/data/sx-mathoverflow/sx-mathoverflow'
# #folder_path='/home/gjq/code_repository/NewWork/data/sx-mathoverflow/sx-mathoverflow_6month'
# # folder_path='/home/gjq/code_repository/NewWork/data/sx-askubuntu/sx-askubuntu_6month'
# # folder_path='/home/gjq/code_repository/NewWork/data/sx-stackoverflow/sx-stackoverflow'



class only_GCN(torch.nn.Module):
    def __init__(self,input_dim,hid_dim, ffn_hidden, n_head, layer_num, drop_prob,label_num,device):
        super().__init__()
        self.hid_dim=hid_dim
        self.device = device
        self.gnn = GCN(input_dim = input_dim, out_dim = self.hid_dim).to(self.device)
        self.simple_linear = nn.Sequential(nn.Linear(hid_dim, label_num), nn.Sigmoid())

    def forward(self,egonet_edges,egonet_x,eval=False):
        _,node_emb=self.gnn(egonet_x.to(self.device),egonet_edges.to(self.device))
        output=self.simple_linear(node_emb.to(self.device))
        return output

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_edges_prd(adjs_list):
    pos_edges_l, false_edges_l = [], []
    edges_list = []
    for i in range(0, len(adjs_list)):
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - \
            sp.dia_matrix(
                (adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = sparse_to_tuple(adj)[0]
        num_false = int(edges.shape[0])

        pos_edges_l.append(edges)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        edges_false = []
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            edges_false.append([idx_i, idx_j])

        assert ~ismember(edges_false, edges_all)

        false_edges_l.append(edges_false)

    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l

def get_roc_scores(edges_pos, edges_neg, adj_orig_dense_list, embs):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Predict on test set of edges
    adj_rec = embs.detach().cpu().numpy()
    adj_orig_t = adj_orig_dense_list
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig_t[e[0], e[1]])
        
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig_t[e[0], e[1]])
    
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    auc_scores=roc_auc_score(labels_all, preds_all)
    ap_scores=average_precision_score(labels_all, preds_all)

    return auc_scores, ap_scores

def _nll_bernoulli(logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / \
            float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(
            input=logits, target=target_adj_dense.to(device), pos_weight=posw, reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0, 1])
        return - nll_loss

def create_egonets(graph,linkprd_edges):
    """
    为pos_edge和false_edege创建并返回其egonet（包含节点自身和一跳邻居的子图）的字典。
    """
    # print(linkprd_edges)
    egonets = {}
    for edge in linkprd_edges:
        first=edge[0]
        second=edge[1]
        # 获取节点的一跳邻居
        neighbors_first = set(graph.neighbors(first))
        neighbors_second = set(graph.neighbors(second))
        neighbors=neighbors_first.union(neighbors_second)
        # 将节点自身加入邻居集合以构成egonet
        egonet_nodes = neighbors.union({first,second})
        # 从原图中提取这些节点及它们之间的边来创建egonet子图
        egonet = graph.subgraph(egonet_nodes).copy()  # 使用.copy()确保得到一个新的图实例
        egonets[(first,second)] = egonet
    return egonets

class data_process:
    def __init__(self,input_dim, device):
        self.device = device
        self.input_dim=input_dim
        self.load_edge_index(folder_path)
        # self.load_feature()

    def load_edge_index(self,folder_path):
        print("加载edge")
        pt_files = [file for file in os.listdir(folder_path) if file.endswith('.pt')]
        pt_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        tensors = []
        for file_name in pt_files:
            file_path = os.path.join(folder_path, file_name)
            # print(file_path)
            tensor = torch.load(file_path)
            tensors.append(tensor)
        # 长度为graph总数的列表，每个graph大小为torch.Size([2, 边数])
        self.edge_list=tensors

    def load_feature(self):
        print("加载feature")
        # spatial_emb = SpatialEncoding(embedding_dim, max_len)
        tensors = []
        for edge_tensor in self.edge_list:
            flattened_edges = edge_tensor.view(-1)
            nodes_set = set(flattened_edges.tolist())
            total_nodes = max(nodes_set)
            tensors.append(torch.ones(total_nodes,self.input_dim))
        self.x_list=tensors

    # 测试阶段不能是egonet，而必须是按节点划分                       
    def split_subgraph(self,seq_len,threshold):
        print(f"graph长度为: {len(self.edge_list)}")
        print(f"seq长度设置为: {seq_len}")
        sub_edges=[]
        for i in range(seq_len):
            mask=(self.edge_list[i][0] < threshold) & (self.edge_list[i][1] < threshold)&(self.edge_list[i][0] != self.edge_list[i][1])
            filtered_edges = self.edge_list[i][:, mask]
            sub_edges.append(filtered_edges)
        return sub_edges

    def make_feature(self,seq_len,node_num):
        return torch.ones(seq_len,node_num,self.input_dim)

def renumber_nodes(egonet_edges):
    for egonet in egonet_edges:
        # 先将原始张量展平
        flat_tensor = egonet.flatten()
        # 使用argsort和cumcount函数来生成连续的索引
        sorted_indices = flat_tensor.argsort()
        unique_values, inverse_indices = flat_tensor[sorted_indices].unique_consecutive(return_inverse=True)
        value_to_index_dict = {int(v): i for i, v in enumerate(unique_values)}
        # 使用查找表得到映射后的索引
        for t,original_tensor in enumerate(egonet):
            mapped_indices = original_tensor.flatten()
            for i in range(mapped_indices.numel()):
                mapped_indices[i] = value_to_index_dict[int(mapped_indices[i])]
            egonet[t]=mapped_indices.view(original_tensor.shape)
    return egonet_edges

def filter_self_loop(edges):
    edges=edges.squeeze(0)
    # 找出自环边的索引
    self_loop_indices = torch.where(edges[0, :] == edges[1, :])[0]

    # 删除自环边
    filtered_edges = torch.cat((edges[:, ~torch.isin(torch.arange(edges.shape[1]), self_loop_indices)],
                               edges[:, self_loop_indices[:0]]), dim=1)
    return filtered_edges[None,:,:]


data_process=data_process(input_dim=input_dim,device=device)
edges=data_process.split_subgraph(seq_len,threshold)
x=data_process.make_feature(seq_len,threshold)


# 防止threshold设置过大，在取false_edge的时候取到孤立结点
max_node=torch.max(edges[seq_end]).item()
print(f'最大节点序号为{max_node}')


adj_time_list=[]
for i in range(seq_len):
    row=np.array(edges[i][0])
    col=np.array(edges[i][1])
    data=np.array(torch.ones_like(edges[i][0]))
    coo_matrix=csr_matrix((data,(row,col)),shape=[threshold, threshold])
    adj_time_list.append(coo_matrix)

adj_orig_dense_list=[]
for i in adj_time_list:
    adj_orig_dense_list.append(torch.tensor(i.todense(),dtype=torch.float32))

pos_edges_l, false_edges_l = mask_edges_prd(adj_time_list)

# 把每个时间片上的pos和false边组合在一起作为all_edges
linkprd_edges=[]
for i in range(len(pos_edges_l)):
    linkprd_edges.append(np.concatenate((pos_edges_l[i], false_edges_l[i]), axis=0))

# 有个问题，原先是预测的graph和真实graph做loss，现在应该是第t时刻的边在t-1时刻的egonet过linear
# 和sigmoid得到边在t时刻的预测类别
G=nx.Graph()
G.add_nodes_from([i for i in range(threshold)])
G.add_edges_from(edges[seq_end-2].t().tolist())
egonets = create_egonets(G,linkprd_edges[seq_end-1])
edge_egonets=[] 
edge_label=[]
edge_egonets_x=[]
for edge in linkprd_edges[seq_end-1]:
    if tuple(edge) in [tuple(i) for i in pos_edges_l[seq_end-1]]:
        edge_label.append(1)
    else:
        edge_label.append(0)
    # 因为前面加false边会引入实际上是孤立的节点，egonets[tuple(edge)].edges()有可能为空，所以需要添加判断
    # if tuple(edge) in egonets.keys() and egonets[tuple(edge)].edges():
    if egonets[tuple(edge)].edges():
        ego_edge=torch.tensor(list(egonets[tuple(edge)].edges()),dtype=torch.int64).t().contiguous()
        edge_egonets.append(ego_edge[None,:,:]) # 扩充维度，一个图测试所以batch_size为1，只用一个时间片所以seq_len为1
        # 有一点就是后面会renumber，nodes()返回的节点序号没有排序，需要排序后提出来，才可以和
        # renumber后的边对应节点对应起来
        indices=list(egonets[tuple(edge)].nodes())
        indices.sort()
        edgegonet_x=x[seq_end-2][indices,:]
        edge_egonets_x.append(edgegonet_x[None,:,:])
    else:
        edge_egonets.append(torch.zeros(1,2,0,dtype=torch.int64))

        # 如果没边，就得把edge的源/目标节点加进去
        # print(torch.tensor(edge))
        edgegonet_x=x[seq_end-2][torch.tensor(edge),:]
        edge_egonets_x.append(edgegonet_x[None,:,:])

# 每个edge的egonet过一遍model
re_edge_egonets=[]
for index in range(len(edge_egonets)):
    filtered_edge=filter_self_loop(edge_egonets[index])
    # 空边没法renumber
    if filtered_edge.shape[2]: 
        re_edge_egonets.append(renumber_nodes(filtered_edge))
    else:
        re_edge_egonets.append(filtered_edge)


# 测试：seq_end-1预测seq_end，用调通的linear测试
G_test=nx.Graph()
G_test.add_nodes_from([i for i in range(threshold)])
G_test.add_edges_from(edges[seq_end-1].t().tolist())
egonets_test = create_egonets(G_test,linkprd_edges[seq_end])
edge_egonets_test=[] 
edge_label_test=[]
edge_egonets_x_test=[]
for edge in linkprd_edges[seq_end]:
    if tuple(edge) in [tuple(i) for i in pos_edges_l[seq_end]]:
        edge_label_test.append(1)
    else:
        edge_label_test.append(0)
    # 因为前面加false边会引入实际上是孤立的节点，egonets[tuple(edge)].edges()有可能为空，所以需要添加判断
    if tuple(edge) in egonets_test.keys() and egonets_test[tuple(edge)].edges():
        ego_edge=torch.tensor(list(egonets_test[tuple(edge)].edges()),dtype=torch.int64).t().contiguous()
        edge_egonets_test.append(ego_edge[None,:,:]) # 扩充维度，一个图测试所以batch_size为1，只用一个时间片所以seq_len为1
        # 有一点就是后面会renumber，nodes()返回的节点序号没有排序，需要排序后提出来，才可以和
        # renumber后的边对应节点对应起来
        indices=list(egonets_test[tuple(edge)].nodes())
        indices.sort()
        edgegonet_x=x[seq_end-1][indices,:]
        edge_egonets_x_test.append(edgegonet_x[None,:,:])
    else:
        edge_egonets_test.append(torch.zeros(1,2,0,dtype=torch.int64))
        # 如果没边，就得把edge的源/目标节点加进去
        edgegonet_x=x[seq_end-1][torch.tensor(edge),:]
        edge_egonets_x_test.append(edgegonet_x[None,:,:])

re_edge_egonets_test=[]
for index in range(len(edge_egonets_test)):
    filtered_edge=filter_self_loop(edge_egonets_test[index])
    if filtered_edge.shape[2]: 
        re_edge_egonets_test.append(renumber_nodes(filtered_edge))
    else:
        re_edge_egonets_test.append(filtered_edge)


model=only_GCN(input_dim=input_dim,hid_dim = hid_dim, ffn_hidden=ffn_hidden, n_head=n_head, layer_num = layer_num, drop_prob=drop_prob,label_num=1,device = device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
criterion = nn.BCELoss()

for i in range(300):
    output=[]
    for index in range(len(edge_egonets)):
        out=model(re_edge_egonets[index].squeeze(0),edge_egonets_x[index].squeeze(0))
        output.append(out[0])
    # print(output)
    loss=criterion(torch.tensor(output, requires_grad=True),torch.FloatTensor(edge_label))
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    output_test=[]
    for index in range(len(re_edge_egonets_test)):
        out_eval=model(re_edge_egonets_test[index].squeeze(0),edge_egonets_x_test[index].squeeze(0))
        output_test.append(out_eval[0])
    label_true=torch.FloatTensor(edge_label_test)
    label_prd=torch.tensor(output_test)
    auc_scores_prd=roc_auc_score(label_true,label_prd)
    ap_scores_prd=average_precision_score(label_true,label_prd)
    print('----------------------------------')
    print('epoch: ', i)
    print(f"loss ={loss:.8f}")
    print('Link Prediction')
    print('link_prd_auc_mean', auc_scores_prd)
    print('link_prd_ap_mean', ap_scores_prd)