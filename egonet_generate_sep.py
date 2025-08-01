import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
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
import hashlib


# pretrain_dataset='/home/ltp/code_repository/NewWork/data/sx-mathoverflow/sx-mathoverflow'
# pretrain_datasets = ['/home/ltp/code_repository/NewWork/data/fb-wosn-friends_train/fb-wosn-friends',
#                     '/home/ltp/code_repository/NewWork/data/ca-cit-HepTh_train/ca-cit-HepTh',
#                     '/home/ltp/code_repository/NewWork/data/sx-stackoverflow_train/sx-stackoverflow_3month',
#                     '/home/ltp/code_repository/NewWork/data/sx-askubuntu_train/sx-askubuntu_3month',
#                     '/home/ltp/code_repository/NewWork/data/ia-prosper-loans_train/ia-prosper-loans',
#                     '/home/ltp/code_repository/NewWork/data/soc-bitcoin_train/soc-bitcoin',
#                     '/home/ltp/code_repository/NewWork/data/ia-enron-email-dynamic_train/ia-enron-email-dynamic']
pretrain_datasets = ['/home/sdc/gjq/code_repository/Pre/data/ca-cit-HepTh_train/ca-cit-HepTh']
def egonet_discovery(center_nodes,edges_seq,node_limit,sample_num):
    egonets_edges=[]
    # print(edges)
    nxGraph_list=[]
    nxnodes_list=[]
    for edges in edges_seq:
        G = nx.Graph(edges.t().tolist())
        nxGraph_list.append(G)
        nodes=G.nodes()
        nxnodes_list.append(nodes)

    for center_node in tqdm(center_nodes):
        isolate_times=0
        egonet_edges=[]
        egonet_x=[]
        for i,edges in enumerate(edges_seq):
            # 因为centernode是从union上取的，不一定出现在每一时间片，
            # 当前节点没出现过，想从该节点找邻居：networkx.exception.NetworkXError: The node 985166 is not in the graph.
            # 先把点加进去，没边就是孤立结点，isolate再判断
            if center_node not in nxnodes_list[i]:
                nxGraph_list[i].add_node(center_node)
            egonet = nx.Graph()
            egonet.add_node(center_node)
            order=0
            while len(egonet.nodes())<node_limit and order<1:
                order+=1
                new_nodes_to_add=set()
                for egonet_node in egonet.nodes():
                    neighbors = list(nxGraph_list[i].neighbors(egonet_node))
                    neighbors_order = [n for n in neighbors if n not in egonet.nodes()]
                    new_nodes_to_add.update(neighbors_order)
                    if len(new_nodes_to_add)>node_limit:
                        break
                egonet.add_nodes_from(list(new_nodes_to_add))
            egonet = nxGraph_list[i].subgraph(egonet.nodes())
            edges_tensor=torch.tensor(list(egonet.edges())).t().contiguous()
            if 0 in edges_tensor.shape:
                isolate_times+=1
                egonet_edges.append(torch.tensor([[center_node],[center_node]]))
            else:
                egonet_edges.append(edges_tensor)
        if isolate_times<len(egonet_edges)/4:
            egonets_edges.append(egonet_edges)
        if len(egonets_edges)>sample_num:
            break
    return egonets_edges #[egonet_num,time,2,edges_num]

class data_process:
    def __init__(self, input_dim, pretrain_dataset,device):
        self.device = device
        self.input_dim = input_dim  
        self.load_edge_index(pretrain_dataset)

    def load_edge_index(self,folder_path):
        print('---------------------------------------------------------')
        print(f"数据集{folder_path.split('/')[-1]}")
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
        self.edge_list=tensors

    # 应当是每个时间片的graph都需要视为大图，同一组但不同时间片的graph的索引一致
    # 子图划分之后可以进行重编号，大图不需要过GCN和attention                           
    def cluster_subgraph(self,seq_len,node_limit):
        print("生成子图划分")
        print(f"graph长度为: {len(self.edge_list)}")
        print(f"seq长度设置为: {seq_len}")
        sample_num=int(2*node_max//(len(self.edge_list)/seq_len))
        egonet_edges=[]
        # sample_num=len(self.edge_list)/seq_len
        for i in range(len(self.edge_list)//seq_len):
            print(f'第{i}次抽序列')
            # union_nodes=set()
            # for j in range(i*seq_len,(i+1)*seq_len):
            #     union_nodes=union_nodes.union(set(self.edge_list[j].reshape(-1).tolist()))
            # selected_elements = random.sample(union_nodes, min(sample_num+50, len(union_nodes)))
            counter = Counter()
            for j in range(i*seq_len,(i+1)*seq_len):
                counter.update(self.edge_list[j].reshape(-1).tolist())
            selected_elements=[element for element, count in counter.most_common(sample_num)]
            edge_seq=egonet_discovery(selected_elements,self.edge_list[i*seq_len:(i+1)*seq_len],node_limit,sample_num) # 长度为time的list:[time,[egonet_num,2,edges_num]]
            # print(center_nodes)
            egonet_edges=egonet_edges+edge_seq
        return egonet_edges

    def renumber_nodes(self,egonet_edges):
        # egonet_x=[]
        egonet_edges_nozero=[]
        for egonet in egonet_edges:
            # 统计所有节点出现频率前 node_max 个
            flat_elements = []
            for tensor in egonet:
                flat_elements.extend(tensor.flatten().tolist())
            element_counter = Counter(flat_elements)
            most_common_elements = element_counter.most_common(node_max)
            value_to_index_dict = {elem: rank for rank, (elem, count) in enumerate(most_common_elements)}
            # 使用查找表得到映射后的索引
            edge_nozero=[]
            for t,original_tensor in enumerate(egonet):
                filtered_edges = []
                for i in range(original_tensor.shape[1]):
                    node1, node2 = original_tensor[:, i]
                    rank1 = value_to_index_dict.get(node1.item(), node_max + 1)
                    rank2 = value_to_index_dict.get(node2.item(), node_max + 1)
                    if rank1 < node_max and rank2 < node_max:
                        filtered_edges.append([rank1, rank2])
                if len(filtered_edges)==0:
                    break
                edge_nozero.append(torch.tensor(filtered_edges).T)
            if len(edge_nozero)==seq_len:
                egonet_edges_nozero.append(edge_nozero)
                # egonet_center_nozero.append(value_to_index_dict.get(center))

            # node_features = torch.tensor(np.eye(node_max).astype(np.float32))
            # # node_features = torch.stack([hash_node_id(node_id) for node_id in node_index])
            # x=[]
            # for t in range(len(egonet)):
            #     x.append(node_features)
            # egonet_x.append(x)
        return egonet_edges_nozero  #,egonet_x

    def filter_self_loop(self,egonets):
        print('共有可用graph{}'.format(len(egonets)))
        for ego in egonets:
            for t in range(len(ego)):
                if(ego[t].shape==torch.Size([0])):
                    continue
                loop_edge=ego[t]
                start_nodes = loop_edge[0]  # 大小为 [num]
                end_nodes = loop_edge[1]
                non_self_loop_indices = start_nodes != end_nodes
                ego[t] = loop_edge[:, non_self_loop_indices]
        return egonets

all_dataset_edges=[]
all_dataset_x=[]
for dataset in pretrain_datasets:
    single_process=data_process(input_dim=input_dim, pretrain_dataset=dataset,device=device)
    egonet_edges=single_process.cluster_subgraph(seq_len,node_limit)
    # print(egonet_center)
    #filtered_edge=single_process.filter_self_loop(egonet_edges)
    egonet_edges=single_process.renumber_nodes(egonet_edges)
    all_dataset_edges=all_dataset_edges+egonet_edges

with open('/home/sdc/gjq/code_repository/Pre/egonet_data/edges_stack_cite.pkl', 'wb') as file:
    pickle.dump(all_dataset_edges, file)
