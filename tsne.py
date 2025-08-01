#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from sklearn.datasets import fetch_openml
# from torch_geometric import nn as tgnn
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
import scipy.sparse as sp
from scipy.linalg import block_diag
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import tarfile
import torch.nn.functional as F
import copy
import time
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.datasets import Planetoid
import networkx as nx
import scipy.io as sio
import torch_scatter
import inspect
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits

import matplotlib.pyplot as plt
import matplotlib.colors 

import copy
import pickle
import os
import time
import random
from sklearn.cluster import KMeans

device = torch.device('cuda:0')
seed = 123
np.random.seed(seed)
with open('/home/gjq/code_repository/dataset/fb/adj_time_list.pickle', 'rb') as handle:
    adj_time_list = pickle.load(handle, encoding='iso-8859-1')
with open('/home/gjq/code_repository/dataset/fb/adj_orig_dense_list.pickle', 'rb') as handle:
    adj_orig_dense_list = pickle.load(handle, encoding='bytes')
with open('embedding_fb.pkl', 'rb')as handle:
    tsne_emb=pickle.load(handle, encoding='bytes')
def list_to_dict(lst):
    dictionary = {}
    lst_set=set(lst)
    for value in lst_set:
        dictionary[value]=[]
    for index, value in enumerate(lst):
        dictionary[value].append(index)
    return dictionary
def comm_spectral_clustering(adj,k):
    adj=adj.numpy()
    print('adj.shape',adj.shape)
    idx_zero = np.all(adj == 0, axis=1)
    idx_zero_indices = np.where(idx_zero)[0]
    adj_l=np.delete(adj,idx_zero_indices,axis=0)
    adj_l=np.delete(adj_l,idx_zero_indices,axis=1)
    print('adj_l.shape',adj_l.shape)

    degreeMatrix = np.sum(adj_l, axis=1)
    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adj_l

    # 计算对称归一化拉普拉斯矩阵
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    symmetric_normalized_laplacian_matrix = np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
    eigval, eigvec = np.linalg.eigh(symmetric_normalized_laplacian_matrix)
    ix = np.argsort(eigval)[0:k]
    H=eigvec[:, ix]
    #sp_kmeans = KMeans(n_clusters=k,n_init=10).fit(H)
    #labels =sp_kmeans.labels_#得到聚类后的每组数据对应的标签类型

    # 使用K均值聚类对特征向量进行聚类
    sp_kmeans = KMeans(n_clusters=k,n_init=10).fit(H)
    labels =sp_kmeans.labels_
    #print(labels.shape)
    pos=0
    isolated=idx_zero
    cluster=[]
    for j in range(len(adj)):
        if isolated[j]==True:
            cluster.append(-1)
        else:
            cluster.append(labels[pos])
            pos+=1
    return list_to_dict(cluster)

tsne = TSNE(n_components=2, perplexity=40, n_iter=300)

tsne_emb_processed=tsne_emb[1]
for i in range(1,len(tsne_emb)):
    tsne_emb_processed=torch.cat([tsne_emb_processed,tsne_emb[i]],1)
embedded_data_1 = tsne.fit_transform(tsne_emb_processed.cpu().detach().numpy())
embedded_data_2 = tsne.fit_transform(tsne_emb[-2].cpu().detach().numpy())

#comm_tsne_list=comm_spectral_clustering(adj_orig_dense_list[-1],6)

ckpt_dir="images"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
num_nodes=adj_orig_dense_list[-1].shape[0]
temp_list=torch.zeros(num_nodes,num_nodes)
for i in range(len(adj_orig_dense_list)):
    temp_list=temp_list+adj_orig_dense_list[i]
comm_tsne_list=comm_spectral_clustering(temp_list,7)


col =  [None]*embedded_data_1.shape[0]
colors = ['green', 'blue', 'red', 'black', 'orange','yellow','pink']
for i in comm_tsne_list.keys():
    for j in comm_tsne_list[i]:
        col[j]=i

frame = plt.gca()
# y 轴不可见
frame.axes.set_yticks([])
# x 轴不可见
frame.axes.set_xticks([])


# print(col)
# print(comm_tsne_list)
# print(col)
#emb_processed=
cm=matplotlib.colors.ListedColormap([(237/255,221/255,195/255),(231/255,56/255,71/255),(217/255,79/255,51/255),(144/255,190/255,224/255),'yellow',(238/255,221/255,195/255)])
plt.scatter(embedded_data_1[:, 0], embedded_data_1[:, 1],c=col,cmap=cm)
#plt.scatter(embedded_data_2[:, 0], embedded_data_2[:, 1],c=col,cmap=cm)
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
        }
plt.xlabel('Dimension 1',font)
plt.ylabel('Dimension 2',font)
plt.savefig('images/digits_tsne_fb.png', dpi=120)
plt.savefig('images/digits_tsne_fb.eps')

plt.show()