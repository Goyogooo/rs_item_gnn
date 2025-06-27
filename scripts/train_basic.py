import os, pickle, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import trange
from sklearn.metrics import ndcg_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument('--no_content', action='store_true', help="不使用 item–sim–item 边 / 内容视图")
parser.add_argument('--no_attr', action='store_true', help="不使用 item–attr 边 / 属性视图")
args = parser.parse_args()

class CIGHCL_Heterarchical(nn.Module):
    def __init__(self, feat_dim, hidden, num_layers, num_users, num_attrs, args):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.args = args

        # 投影 & 嵌入
        self.lin_item = nn.Linear(feat_dim, hidden)
        self.user_emb = nn.Embedding(num_users, hidden)
        self.attr_emb = nn.Embedding(num_attrs, hidden)

        # 局部层
        self.convs_ui = nn.ModuleList([
            HeteroConv({
                ('user','rates','item'):     SAGEConv((hidden,hidden), hidden),
                ('item','rev_rates','user'): SAGEConv((hidden,hidden), hidden),
                ('user','self','user'):      SAGEConv(hidden, hidden),
                ('item','self','item'):      SAGEConv(hidden, hidden),
            }, aggr='mean')
            for _ in range(num_layers)
        ])

        # 内容视图多层
        if not args.no_content:
            self.convs_ii = nn.ModuleList([
                HeteroConv({
                    ('item','sim','item'):   SAGEConv(hidden, hidden),
                    ('item','self','item'):  SAGEConv(hidden, hidden),
                }, aggr='mean')
                for _ in range(num_layers)
            ])
        else:
            self.convs_ii = None

        # 属性视图多层：item↔attr
        if not args.no_attr:
            self.convs_ia = nn.ModuleList([
                HeteroConv({
                    ('item','has','attr'):      SAGEConv((hidden,hidden), hidden),
                    ('attr','rev_has','item'):  SAGEConv((hidden,hidden), hidden),
                    ('item','self','item'):     SAGEConv(hidden, hidden),
                    ('attr','self','attr'):     SAGEConv(hidden, hidden),
                }, aggr='mean')
                for _ in range(num_layers)
            ])
        else:
            self.convs_ia = None

        self.alpha = nn.Parameter(torch.zeros(3))
        
        # 对比投影头
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, data):
        x_item = self.lin_item(data['item'].x)
        x_user = self.user_emb.weight
        x_attr = self.attr_emb.weight
        edges  = data.edge_index_dict

        # 交互 & 内容 & 属性局部聚合
        h_ui = {'user':x_user, 'item':x_item}
        for conv in self.convs_ui:
            h_ui = conv(h_ui, edges)
            h_ui = {k:F.relu(v) for k,v in h_ui.items()}
        h_ui_item = h_ui['item']

        if self.convs_ii is not None:
            h_ii = {'item': x_item}
            for conv in self.convs_ii:
                h_ii = conv(h_ii, edges)
                h_ii = {k: F.relu(v) for k,v in h_ii.items()}
            h_ii_item = h_ii['item']
        else:
            h_ii_item = torch.zeros_like(h_ui_item)

        if self.convs_ia is not None:
            h_ia = {'item': x_item, 'attr': x_attr}
            for conv in self.convs_ia:
                h_ia = conv(h_ia, edges)
                h_ia = {k: F.relu(v) for k,v in h_ia.items()}
            h_ia_item = h_ia['item']
        else:
            h_ia_item = torch.zeros_like(h_ui_item)

        # —— 跨视图层：可学习门控融合
        w = torch.softmax(self.alpha, dim=0)
        h_fused = w[0]*h_ui_item + w[1]*h_ii_item + w[2]*h_ia_item

        return h_fused, h_ui_item, h_ii_item, h_ia_item, w

    def loss_contrast(self, hi, hc):
        zi = self.proj(hi); zc = self.proj(hc)
        sim = zi@zc.t() / (zi.norm(1)*zc.norm(1).t())
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)
    

# ========== 超参 ==========
LAMBDA     = 0.2    # 对比损失权重
WEIGHT_DEC = 1e-4   # Adam 权重衰减
LR         = 1e-3
HIDDEN     = 64
LAYERS     = 2
EPOCHS     = 60
BATCH_SIZE = 4096
K_NEG      = 99    # 负采样数
K_TOP      = 10    # 推荐 Top-K

CACHE      = "data/ml-1m/cache"


def bpr_loss(u_emb, pos_emb, neg_emb):
    pos = (u_emb * pos_emb).sum(dim=1)
    neg = (u_emb * neg_emb).sum(dim=1)
    return -F.logsigmoid(pos - neg).mean()


def leave_one_evaluate(model, data, pairs_n, k=K_TOP, num_neg=K_NEG):
    """对 (user_n,item_n) 列表评 HR/NDCG@k"""
    model.eval()
    with torch.no_grad():
        h_fused, _, _, _,_ = model(data)
    H = h_fused.cpu().numpy()
    train_set = set(map(tuple, data['user','rates','item'].edge_index.t().cpu().tolist()))

    hits, ndcgs, mrrs = [], [], []
    for u_n, pos_i in pairs_n:
        negs = []
        while len(negs) < num_neg:
            ni = np.random.randint(0, H.shape[0])
            if (u_n, ni) not in train_set and ni != pos_i:
                negs.append(ni)
        items = negs + [pos_i]
        scores = (H[items] * H[pos_i]).sum(axis=1)
        labels = np.zeros(len(items), dtype=int); labels[-1] = 1

        # HR & NDCG
        idx = np.argsort(scores)[-k:]
        hits.append(int((len(items)-1) in idx))
        ndcgs.append(ndcg_score([labels], [scores], k=k))

        # MRR
        sorted_idx = np.argsort(scores)[::-1]
        for rank, i in enumerate(sorted_idx, 1):
            if labels[i] == 1:
                mrrs.append(1.0 / rank)
                break

    return np.mean(hits), np.mean(ndcgs), np.mean(mrrs)


def train():
    # —— 1) 加载图（基于 ui_train 构建）
    with open(f"{CACHE}/graph_cighcl_basic_pyg.bin", "rb") as f:
        data: HeteroData = pickle.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # —— 2) 读取 train/valid 划分
    train_df = pd.read_csv(f"{CACHE}/ui_train.csv")  # 原 uid, mid
    valid_df = pd.read_csv(f"{CACHE}/ui_valid.csv")

    # 重建映射：确保和 build_graph 时一致
    unique_u = sorted(train_df.uid.unique())
    item_idx = pd.read_csv(f"{CACHE}/item_idx.csv")  # 原 mid 列
    unique_i = sorted(item_idx.mid.unique())
    u2nid    = {u:i for i,u in enumerate(unique_u)}
    i2nid    = {i:idx for idx,i in enumerate(unique_i)}

    # 映射 train
    train_df['uid_n'] = train_df.uid.map(u2nid)
    train_df['mid_n'] = train_df.mid.map(i2nid)
    train_pairs_n     = train_df[['uid_n','mid_n']].values.tolist()

    # 映射 valid
    valid_df['uid_n'] = valid_df.uid.map(u2nid)
    valid_df['mid_n'] = valid_df.mid.map(i2nid)
    valid_pairs_n     = valid_df[['uid_n','mid_n']].values.tolist()

    # —— 3) 构建正负样本字典
    pos_dict = {}
    for u_n, i_n in train_pairs_n:
        pos_dict.setdefault(u_n, []).append(i_n)
    all_items = list(range(data['item'].num_nodes))
    users     = list(pos_dict.keys())

    # —— 4) 初始化模型 & 优化器
    num_users = data['user'].num_nodes
    num_items = data['item'].num_nodes
    num_attrs = data['attr'].num_nodes
    feat_dim  = data['item'].x.size(1)

    model = CIGHCL_Heterarchical(
        feat_dim, HIDDEN, LAYERS,
        num_users=num_users, num_attrs=num_attrs, args=args
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DEC
    )


    best_hr = 0.0
    patience = 5  # 设置 patience 值，即允许的最大没有提升的 epoch 数
    no_improvement_count = 0  # 用于计数没有提升的 epoch 数
    # —— 5) 训练循环
    for epoch in trange(1, EPOCHS+1, desc="Training"):
        model.train()
        
        total_loss = 0.0
        random.shuffle(users)

        for start in range(0, len(users), BATCH_SIZE):
            batch = users[start:start+BATCH_SIZE]
            u_batch, pos, neg = [], [], []
            for u in batch:
                cand = pos_dict.get(u, [])
                if not cand:
                    continue
                u_batch.append(u)
                pos.append(random.choice(cand))
                neg.append(random.choice(all_items))
            if not u_batch:
                continue
            

            h_fused, h_ui, h_ii, h_ia , w= model(data)
            view_weights.append(w.detach().cpu().numpy())
            optimizer.zero_grad()
            

            # BPR 损失
            u_emb   = model.user_emb(torch.tensor(u_batch, device=device))
            pos_emb = h_fused[pos]
            neg_emb = h_fused[neg]
            loss_bpr = bpr_loss(u_emb, pos_emb, neg_emb)

            # 对比损失
            loss_c1 = model.loss_contrast(h_ui, h_ii)
            loss_c2 = model.loss_contrast(h_ui, h_ia)
            loss_con = LAMBDA * (loss_c1 + loss_c2)

            loss = loss_bpr + loss_con
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(u_batch)

        avg_loss = total_loss / len(users)
        hr, ndcg, mrr = leave_one_evaluate(model, data, valid_pairs_n)

        print(f"Epoch {epoch:02d} | Loss={avg_loss:.4f} | HR@{K_TOP}={hr:.4f}, NDCG@{K_TOP}={ndcg:.4f}, MRR@{K_TOP}={mrr:.4f}")


        # 早停：保存最优模型
        if hr > best_hr:
            best_hr = hr
            torch.save(model.state_dict(), "models/best_cighcl.pt")
            no_improvement_count = 0
        else:
            print("  (no improvement)")
            no_improvement_count += 1
        
        if no_improvement_count >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 最终模型替换
    os.makedirs("models", exist_ok=True)
    os.replace("models/best_cighcl.pt", "models/cighcl_basic_pyg.pt")
    print(" Training complete, best model saved.")



def evaluate_test():
    # 加载保存的最优模型
    with open(f"{CACHE}/graph_cighcl_basic_pyg.bin", "rb") as f:
        data: HeteroData = pickle.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # 重新建立映射
    train_df = pd.read_csv(f"{CACHE}/ui_train.csv")
    test_df  = pd.read_csv(f"{CACHE}/ui_test.csv")
    unique_u = sorted(train_df.uid.unique())
    item_idx = pd.read_csv(f"{CACHE}/item_idx.csv")
    unique_i = sorted(item_idx.mid.unique())
    u2nid    = {u:i for i,u in enumerate(unique_u)}
    i2nid    = {i:idx for idx,i in enumerate(unique_i)}

    # 测试集映射
    test_df['uid_n'] = test_df.uid.map(u2nid)
    test_df['mid_n'] = test_df.mid.map(i2nid)
    test_pairs_n     = test_df[['uid_n','mid_n']].dropna().astype(int).values.tolist()

    # 初始化模型
    model = CIGHCL_Heterarchical(
        feat_dim  = data['item'].x.size(1),
        hidden    = HIDDEN,
        num_layers= LAYERS,
        num_users = data['user'].num_nodes,
        num_attrs = data['attr'].num_nodes,
        args=args
    ).to(device)

    model.load_state_dict(torch.load("models/cighcl_basic_pyg.pt"))
    model.eval()

    hr, ndcg, mrr = leave_one_evaluate(model, data, test_pairs_n)
    print(f"[Test Set] HR@{K_TOP} = {hr:.4f}, NDCG@{K_TOP} = {ndcg:.4f}, MRR@{K_TOP} = {mrr:.4f}")


if __name__ == "__main__":

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train()
    evaluate_test() 