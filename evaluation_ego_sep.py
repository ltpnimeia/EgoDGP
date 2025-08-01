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
# from torch.nn.functional import cosine_similarity
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
from tqdm import tqdm, trange
from tqdm.contrib import tzip
from collections import Counter
import random
from model import DGPretrain
from config import *
import hashlib

torch.set_printoptions(profile="full")

threshold = 1000
seq_len = 1
seq_start = 20
seq_end = seq_start+seq_len
# folder_path='/home/ltp/code_repository/NewWork/data/sx-mathoverflow/sx-mathoverflow'
# folder_path='/home/ltp/code_repository/NewWork/data/sx-stackoverflow_train/sx-stackoverflow_3month'
# folder_path='/home/ltp/code_repository/NewWork/data/ca-cit-HepTh_train/ca-cit-HepTh'
# folder_path='/home/ltp/code_repository/NewWork/data/sx-mathoverflow/sx-mathoverflow'
# folder_path='/home/ltp/code_repository/NewWork/data/sx-askubuntu_train/sx-askubuntu_3month'
# folder_path='/home/ltp/code_repository/NewWork/data/ia-enron-email-dynamic_train/ia-enron-email-dynamic'
#folder_path = '/home/ltp/code_repository/NewWork/data/fb-wosn-friends_train/fb-wosn-friends'
#                     '/home/ltp/code_repository/NewWork/data/ca-cit-HepTh_train/ca-cit-HepTh'
#                     '/home/ltp/code_repository/NewWork/data/sx-stackoverflow_train/sx-stackoverflow_3month'
#                     '/home/ltp/code_repository/NewWork/data/sx-askubuntu_train/sx-askubuntu_3month'
#                     '/home/ltp/code_repository/NewWork/data/ia-prosper-loans_train/ia-prosper-loans'
# folder_path='/home/ltp/code_repository/NewWork/data/soc-bitcoin_train/soc-bitcoin'
# folder_path='/home/ltp/code_repository/NewWork/data/fb-wosn-friends_train/fb-wosn-friends'
# folder_path='/home/ltp/code_repository/NewWork/data/ia-enron-email-dynamic_train/ia-enron-email-dynamic'

#                     '/home/ltp/code_repository/NewWork/data/ia-enron-email-dynamic_train/ia-enron-email-dynamic'
pretrain_model = 'pretarin_dict_sep_mix_bigger.pt'

def adjacency_matrices_to_edge_lists(graph_list):
    """
    将一系列邻接矩阵转换为边列表。
    
    参数:
    graph_list (list of csr_matrix): 图的邻接矩阵列表。
    
    返回:
    list of torch.Tensor: 每个图的边列表，每个列表项是一个 [2, edge_num] 形状的张量。
    """
    print('graph长度为：',len(graph_list))
    edges_list = []
    for adj_matrix in graph_list:
        # 获取非零元素的位置
        row_indices, col_indices = adj_matrix.nonzero()
        
        # 将非零元素的位置转换为边
        edges = np.vstack((row_indices, col_indices))
        
        # 如果是无向图，确保边不会重复出现两次
        if adj_matrix.shape[0] == adj_matrix.shape[1]:  # 确保是方阵
            # 移除对角线上的边（如果有）
            mask = row_indices != col_indices
            edges = edges[:, mask]
            
            # 如果是无向图，只需要上三角部分的边
            upper_triangular_mask = row_indices < col_indices
            edges = edges[:, upper_triangular_mask]
        
        # 转换成Tensor
        edges_tensor = torch.tensor(edges, dtype=torch.long)
        
        # 添加到列表中
        edges_list.append(edges_tensor)
    
    return edges_list

def egonet_discovery(center_nodes, edges_seq, node_limit):
    egonets_edges = []
    egonets_center = []
    # print(edges)
    nxGraph_list = []
    nxnodes_list = []
    for edges in edges_seq:
        G = nx.Graph(edges.t().tolist())
        nxGraph_list.append(G)
        nodes = G.nodes()
        nxnodes_list.append(nodes)

    for center_node in tqdm(center_nodes):
        isolate_times = 0
        egonet_edges = []
        egonet_x = []
        for i, edges in enumerate(edges_seq):
            # 因为centernode是从union上取的，不一定出现在每一时间片，
            # 当前节点没出现过，想从该节点找邻居：networkx.exception.NetworkXError: The node 985166 is not in the graph.
            # 先把点加进去，没边就是孤立结点，isolate再判断
            if center_node not in nxnodes_list[i]:
                nxGraph_list[i].add_node(center_node)
            egonet = nx.Graph()
            egonet.add_node(center_node)
            order = 0
            while len(egonet.nodes()) < node_limit and order < 1:
                order += 1
                new_nodes_to_add = set()
                for egonet_node in egonet.nodes():
                    neighbors = list(nxGraph_list[i].neighbors(egonet_node))
                    neighbors_order = [
                        n for n in neighbors if n not in egonet.nodes()]
                    new_nodes_to_add.update(neighbors_order)
                    if len(new_nodes_to_add) > node_limit:
                        break
                egonet.add_nodes_from(list(new_nodes_to_add))
            egonet = nxGraph_list[i].subgraph(egonet.nodes())
            edges_tensor = torch.tensor(list(egonet.edges())).t().contiguous()
            if 0 in edges_tensor.shape:
                isolate_times += 1
                egonet_edges.append(torch.tensor(
                    [[center_node], [center_node]]))
            else:
                egonet_edges.append(edges_tensor)
        egonets_edges.append(egonet_edges)
        egonets_center.append(center_node)
    return egonets_edges, egonets_center

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def generate_samples_from_tensors(edge_tensors):
    """
    为每个图生成正样本和负样本。

    :param edge_tensors: 包含多个图的列表，每个图用形状为 [2, edges_num] 的张量表示。
    :return: 一个包含每个图的正样本和负样本的列表。
    """
    all_samples = []

    for edges in edge_tensors:
        print(edges.shape)
        # 计算最大的节点编号加一作为nodes_num
        nodes_num = int(torch.max(edges)) + 1

        # 正样本数量为边的数量
        pos_edges_per_graph = edges.size(1)

        # 负样本数量与正样本数量相同
        neg_edges_per_graph = pos_edges_per_graph

        # 正样本直接使用现有的边张量，并转换为列表格式
        positive_samples = [edge.tolist() for edge in edges.t()]

        # 生成负样本
        negative_samples = []
        existing_edges = set(map(tuple, positive_samples))

        # 生成负样本，直到达到所需的数量
        while len(negative_samples) < neg_edges_per_graph:
            u_prime, v_prime = random.sample(range(nodes_num), 2)
            if (u_prime, v_prime) not in existing_edges and (v_prime, u_prime) not in existing_edges:
                negative_samples.append([u_prime, v_prime])
                existing_edges.add((u_prime, v_prime))

        # 合并正样本和负样本
        merged_samples = positive_samples + negative_samples

        # 添加到结果列表中
        all_samples.append({
            'positive': positive_samples,
            'negative': negative_samples,
            'merged': merged_samples
        })

    return all_samples

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
    auc_scores = roc_auc_score(labels_all, preds_all)
    ap_scores = average_precision_score(labels_all, preds_all)

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

class data_process:
    def __init__(self, input_dim, device):
        self.device = device
        self.input_dim = input_dim
        self.load_edge_index(folder_path)
        # self.load_feature()

    def load_edge_index(self, folder_path):
        print("加载edge")
        pt_files = [file for file in os.listdir(
            folder_path) if file.endswith('.pt')]
        pt_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        tensors = []
        for file_name in pt_files:
            file_path = os.path.join(folder_path, file_name)
            # print(file_path)
            tensor = torch.load(file_path)
            tensors.append(tensor)
        # 长度为graph总数的列表，每个graph大小为torch.Size([2, 边数])
        self.edge_list = tensors

    # 测试阶段不能是egonet，而必须是按节点划分
    def get_graph(self):
        print(f"graph长度为: {len(self.edge_list)}")
        print(f"seq长度设置为: {seq_len}")
        sub_edges = []
        for i in range(seq_start, seq_end):
            mask = (self.edge_list[i][0] < threshold) & (self.edge_list[i][1] < threshold) & (
                self.edge_list[i][0] != self.edge_list[i][1])
            filtered_edges = self.edge_list[i][:, mask]
            sub_edges.append(filtered_edges)
        return sub_edges
        # return self.edge_list

def renumber_nodes(egonet_edges):
    egonet_edges_nozero = []
    # seq_len_eval=len(egonet_edges[0])
    for egonet in egonet_edges:
        flat_elements = []
        for tensor in egonet:
            flat_elements.extend(tensor.flatten().tolist())
        element_counter = Counter(flat_elements)
        most_common_elements = element_counter.most_common(node_max)
        value_to_index_dict = {
            elem: rank for rank, (elem, count) in enumerate(most_common_elements[:-1])}
        # value_to_index_dict[center]=1
        # print(value_to_index_dict)
        # 使用查找表得到映射后的索引
        edge_nozero = []
        for t, original_tensor in enumerate(egonet):
            filtered_edges = []
            for i in range(original_tensor.shape[1]):
                node1, node2 = original_tensor[:, i]
                rank1 = value_to_index_dict.get(node1.item(), node_max + 1)
                rank2 = value_to_index_dict.get(node2.item(), node_max + 1)
                if rank1 < node_max and rank2 < node_max:
                    filtered_edges.append([rank1, rank2])
            if len(filtered_edges) == 0:
                filtered_edges.append([0, 0])
            edge_nozero.append(torch.tensor(filtered_edges).T)
        # print(len(edge_nozero))
        # if len(edge_nozero)==seq_len_eval:
        egonet_edges_nozero.append(edge_nozero)
        # node_features = torch.stack([hash_node_id(node_id) for node_id in node_index])
        # for t in range(len(egonet)):
        #     x.append(node_features)
        # egonet_x.append(x)
    return egonet_edges_nozero

def filter_self_loop(egonets):
    print('共有可用graph{}'.format(len(egonets)))
    for ego in egonets:
        for t in range(len(ego)):
            if(ego[t].shape == torch.Size([0])):
                continue
            loop_edge = ego[t]
            start_nodes = loop_edge[0]  # 大小为 [num]
            end_nodes = loop_edge[1]
            non_self_loop_indices = start_nodes != end_nodes
            ego[t] = loop_edge[:, non_self_loop_indices]
    return egonets

def cosine_similarity(tensor1, tensor2):
    # Normalize the tensors along the feature dimension (dim=1)
    tensor1_normalized = torch.nn.functional.normalize(tensor1, p=2, dim=1)
    tensor2_normalized = torch.nn.functional.normalize(tensor2, p=2, dim=1)

    # Compute the dot product of the normalized tensors
    similarity_scores = torch.sum(
        tensor1_normalized * tensor2_normalized, dim=1)

    return similarity_scores

class LinkPrediction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinkPrediction, self).__init__()
        # 单一线性层，将输入映射到输出
        self.fc1 = nn.Linear(input_dim, input_dim).to(device)  # 输入层到隐藏层
        self.fc2 = nn.Linear(input_dim, output_dim).to(device)  # 隐藏层到输出层
        # self.simple_linear = nn.Sequential(nn.Linear(hid_dim, 16), nn.ReLU(),nn.Linear(16, output_dim),nn.Sigmoid())
        # self.simple_linear = nn.Linear(hid_dim, output_dim)

    def forward(self, emb0,emb1,emb2):
        # x = F.relu(self.fc1(torch.cat([emb0, emb1, emb2], dim=-1)))
        # x = torch.sigmoid(self.fc2(x))
        # emb1 = F.relu(self.fc1(emb1))
        # emb1 = torch.sigmoid(self.fc2(emb1))
        # emb2 = F.relu(self.fc1(emb2))
        # e mb2 = torch.sigmoid(self.fc2(emb2))
        # print(emb1[2])
        # print(emb2[2])
        emb0 = self.fc1(emb0)
        emb0 = F.relu(self.fc2(emb0))
        emb1 = self.fc1(emb1)
        emb1 = F.relu(self.fc2(emb1))
        emb2 = self.fc1(emb2)
        emb2 = F.relu(self.fc2(emb2))
        # print("houmian")
        # print(emb1[2])
        # print(emb2[2])
        isolate_sim = (torch.abs(emb1-emb0)+torch.abs(emb2-emb0))/2
        isolate_sim = torch.mean(isolate_sim, dim=1)
        node_sim = (cosine_similarity(emb1-emb0, emb2-emb0)+1) / 2
        si_matrix = isolate_sim*node_sim
        si_matrix=torch.sigmoid(si_matrix)
        return si_matrix

# with open('/home/sdc/gjq/code_repository/Pre/data/highSchool_test/adj_time_list.pickle', 'rb') as handle:
#     graph_list  = pickle.load(handle, encoding='iso-8859-1')

# edges = adjacency_matrices_to_edge_lists(graph_list)

folder_path='/home/sdc/gjq/code_repository/Pre/data/ca-cit-HepTh_train/ca-cit-HepTh'
data_process = data_process(input_dim=input_dim, device=device)
edges = data_process.get_graph()


all_samples = generate_samples_from_tensors(edges)

# 把每个时间片上的pos和false边组合在一起作为all_edges
# 最后一个时间片(seq_len-1)上的需要预测的边作为linkprd_edges
random_index = torch.randperm(len(all_samples[seq_len-1]['merged']))
linkprd_edges = [all_samples[seq_len-1]['merged'][index]
                 for index in random_index]
print('正负样例采样完成')
edge_label = []
for edge in linkprd_edges:
    if tuple(edge) in [tuple(i) for i in all_samples[seq_len-1]['positive']]:
        edge_label.append(1)
    else:
        edge_label.append(0)

# [prd_edge,seq_len,edges]
# 根据linkprd_edges,找到edges在[:seq_len-1]="0,1,2,...seq_len-2"上的egonet（不能有(seq_len-1)）

egonet_edges_first, egonet_center_first = egonet_discovery(
    [ele[0] for ele in linkprd_edges], edges[:seq_len-1], node_limit)
egonet_edges_first = renumber_nodes(egonet_edges_first)
egonet_edges_second, egonet_center_second = egonet_discovery(
    [ele[1] for ele in linkprd_edges], edges[:seq_len-1], node_limit)
egonet_edges_second = renumber_nodes(egonet_edges_second)

print('egonet采样完成')
egonet_x = []
for egonet in egonet_edges_first:
    x = []
    node_features = torch.tensor(np.eye(node_max).astype(np.float32))
    for t in range(len(egonet)):
        x.append(node_features)
    egonet_x.append(x)

model = DGPretrain(input_dim=node_max, hid_dim=hid_dim, ffn_hidden=ffn_hidden,
                   n_head=n_head, layer_num=layer_num, drop_prob=drop_prob, device=device).to(device)
model.load_state_dict(torch.load(
    pretrain_model, map_location="cuda:1" if torch.cuda.is_available() else "cpu"))
model.eval()

# LinkPrediction=LinkPrediction(2*hid_dim, 1).to(device)
# optimizer = torch.optim.Adam(LinkPrediction.parameters(), lr=lr, weight_decay=decay)
# criterion = nn.BCELoss()

# print(egonet_edges)
base_tensor = torch.tensor([[0], [0]])
isolated_tensor = base_tensor.unsqueeze(
    0).unsqueeze(0).repeat(1, seq_len-1, 1, 1)

print('进入model')

with torch.no_grad():
    emb_null, _ = model(isolated_tensor, [egonet_x[0]], 'link_prd')
    emb_re_first, _ = model(egonet_edges_first, egonet_x, 'link_prd')
    emb_re_second, _ = model(egonet_edges_second, egonet_x, 'link_prd')

adapter = LinkPrediction(hid_dim, hid_dim).to(device)
optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001, weight_decay=decay)
criterion = nn.CosineSimilarity(dim=-1, eps=1e-6)
# print(emb_re_first[:,-1,:])
# for i in range(20):
#     train_num=4*len(emb_re_first)//5
#     test_num=len(emb_re_first)-4*len(emb_re_first)//5
#     optimizer.zero_grad()
#     emb_null_train = emb_null[0, -1, :].unsqueeze(0).repeat(train_num, 1)
#     output=adapter(emb_null_train,emb_re_first[:train_num,-1,:],emb_re_second[:train_num,-1,:])
#     # print(node_sim)
#     # output=dem1.squeeze()
#     # output=isolate_sim
#     label=torch.FloatTensor(edge_label)
#     # print(output)
#     # print(label)
#     loss=0
#     for g in range(train_num):
#         loss+=(1-criterion(output[g].to(device),label[g].to(device)))
#     # print(loss)
#     loss.backward(retain_graph=True)
#     optimizer.step()
#     emb_null_test = emb_null[0, -1, :].unsqueeze(0).repeat(test_num, 1)
#     output_test=adapter(emb_null_test,emb_re_first[train_num:,-1,:],emb_re_second[train_num:,-1,:])
#     auc_scores_prd = roc_auc_score(edge_label[train_num:], output_test.cpu().detach().numpy())
#     ap_scores_prd = average_precision_score(edge_label[train_num:], output_test.cpu().detach().numpy())
#     print('----------------------------------')
#     print('epoch: ', i)
#     print(f"loss ={loss:.8f}")
#     print('Link Prediction')
#     print('link_prd_auc_mean', auc_scores_prd)
#     print('link_prd_ap_mean', ap_scores_prd)

emb_null = emb_null[0, -1, :].unsqueeze(0).repeat(len(emb_re_first), 1)
isolate_sim = (torch.abs(emb_re_first[:,-1,:]-emb_null)+torch.abs(emb_re_second[:,-1,:]-emb_null))/2
# isolate_sim=torch.abs(emb_re_first[:,-1,:]-emb_null)
isolate_sim = torch.mean(isolate_sim, dim=1)
# print(isolate_sim)
# dem1=LinkPrediction(torch.cat((emb_re_first[:,-1,:],emb_re_second[:,-1,:]),dim=1))
node_sim = (cosine_similarity(
    emb_re_first[:,-1,:]-emb_null, emb_re_second[:,-1,:]-emb_null)+1) / 2
# print(node_sim)
si_matrix = isolate_sim*node_sim
output = si_matrix
# output=dem1.squeeze()
# output=isolate_sim
# loss=criterion(si_matrix,torch.FloatTensor(edge_label).to(device))
# loss.backward(retain_graph=True)
# optimizer.step()
auc_scores_prd = roc_auc_score(edge_label, output.cpu().detach().numpy())
ap_scores_prd = average_precision_score(
    edge_label, output.cpu().detach().numpy())
print('----------------------------------')
# print('epoch: ', i)
# print(f"loss ={loss:.8f}")
print('Link Prediction')
print('link_prd_auc_mean', auc_scores_prd)
print('link_prd_ap_mean', ap_scores_prd)