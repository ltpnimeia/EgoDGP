import torch

seq_len=10   # 进行预训练的seq长度
node_limit=20    # egonet节点数量限制
batch_size=300
lr=0.01
decay=0.0001
epochs=1000
patient=10

node_max=1000
input_dim=64
hid_dim = 128
ffn_hidden=512
n_head=8
layer_num = 8
drop_prob=0

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
