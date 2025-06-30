import os
import pickle

import numpy as np
import pandas as pd
import faiss
import torch
from torch_geometric.data import HeteroData



cache     = "data/ml-1m/cache"
ui_path   = f"{cache}/ui_train.csv"
item_idx  = f"{cache}/item_idx.csv"
feat_path = f"{cache}/item_feat.npy"

# 1) 读交互（训练集）和原始 item 索引
ui      = pd.read_csv(ui_path)         
mid_idx = pd.read_csv(item_idx)         
movies  = pd.read_csv(
    "data/ml-1m/movies.dat",
    sep="::", engine="python",
    names=["mid","title","genres"],
    encoding="latin-1"
)

# 2) 生成连号映射
unique_u = sorted(ui.uid.unique())
unique_i = sorted(mid_idx.mid.unique())
u2nid    = {u: idx for idx, u in enumerate(unique_u)}# 生成用户id到节点id字典
i2nid    = {i: idx for idx, i in enumerate(unique_i)}# 生成电影id到节点id字典
num_users, num_items = len(u2nid), len(i2nid)

# 3) 重映射交互
ui['uid_n'] = ui.uid.map(u2nid)# 交互数据中用户id映射对应节点id
ui['mid_n'] = ui.mid.map(i2nid)# 交互数据中电影id映射对应节点id

# 4) 对齐 item_feat 到新的 节点ID（电影id映射）
feat_all = np.load(feat_path)          # shape = (原 num_items, feat_dim)
mid_idx['mid_n'] = mid_idx.mid.map(i2nid)# 电影特征数据中电影id映射到对应节点id
mid_idx = mid_idx.sort_values('mid_n').reset_index(drop=True)
feat     = feat_all[mid_idx.index]     # shape = (num_items, feat_dim)

# 5) 构建内容相似边（item–sim–item）
k = 20
index = faiss.IndexFlatIP(feat.shape[1])
index.add(feat.astype('float32'))
_, nbr = index.search(feat.astype('float32'), k+1)
# nbr 是一个形状为 (num_items, k+1) 的数组，
# 其中每行包含 每个物品最相似的 k+1 个物品的索引。
# 第一列通常是物品本身（即与自己最相似），接下来的 k 列是最相似的 k 个物品。
ii_src = np.repeat(np.arange(num_items), k)# 源节点num_items * k
ii_dst = nbr[:, 1:].reshape(-1)# 相似物品节点num_items * k

# 6) 构建属性视图（item–has–attr, attr–rev_has–item）
all_genres = sorted({g for gs in movies.genres.str.split("|") for g in gs}) #  电影的类别字符串列表，按字母排序
attr2nid   = {g: i for i, g in enumerate(all_genres)} #  将每个电影类型映射到节点ID，形成字典
num_attrs  = len(all_genres)

edges_ia_src = [] # 保存源节点，物品节点Id
edges_ia_dst = [] # 保存目的节点，属性节点Id
for _, row in movies.iterrows():
    orig_mid = row.mid
    if orig_mid not in i2nid:
        continue
    item_nid = i2nid[orig_mid]
    for g in row.genres.split("|"):
        if g in attr2nid:
            edges_ia_src.append(item_nid)
            edges_ia_dst.append(attr2nid[g])

# 7) 构造 HeteroData
data = HeteroData()
data['user'].num_nodes  = num_users
data['item'].x          = torch.from_numpy(feat).float()
data['attr'].num_nodes  = num_attrs

# 用户→物品 边
src = torch.LongTensor(ui.uid_n.values)
dst = torch.LongTensor(ui.mid_n.values)
data['user', 'rates',  'item'].edge_index     = torch.stack([src, dst], dim=0)
data['item', 'rev_rates', 'user'].edge_index  = torch.stack([dst, src], dim=0)# 物品→用户

# 物品→物品 相似边
data['item', 'sim', 'item'].edge_index = torch.stack([
    torch.LongTensor(ii_src),
    torch.LongTensor(ii_dst)
], dim=0)

# 物品→属性 边
data['item','has','attr'].edge_index     = torch.stack([
    torch.LongTensor(edges_ia_src),
    torch.LongTensor(edges_ia_dst)
], dim=0)
data['attr','rev_has','item'].edge_index = torch.stack([# 属性→物品
    torch.LongTensor(edges_ia_dst),
    torch.LongTensor(edges_ia_src)
], dim=0)

for ntype in ['user','item','attr']:
    n = data[ntype].num_nodes
    idx = torch.arange(n, dtype=torch.long)
    data[ntype, 'self', ntype].edge_index = torch.stack([idx, idx], dim=0)

# 8) 保存
os.makedirs(cache, exist_ok=True)
with open(f"{cache}/graph_cighcl_basic_pyg.bin", 'wb') as f:
    pickle.dump(data, f)

print(f"Built HeteroData with {num_users} users, "
      f"{num_items} items, {num_attrs} attrs.")
