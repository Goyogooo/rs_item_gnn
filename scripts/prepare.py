
import os
import pandas as pd
import numpy as np

ROOT  = "data/ml-1m"
CACHE = f"{ROOT}/cache"
os.makedirs(CACHE, exist_ok=True)

# 1) Ratings（二元化正反馈：rating>=3）
ratings = pd.read_csv(
    f"{ROOT}/ratings.dat",
    sep="::",
    engine="python",
    names=["uid","mid","rate","ts"],
    encoding="latin-1"
)
# 只保留 >=3 的正反馈，并按时间戳排序
ratings = ratings[ratings.rate >= 3].sort_values("ts").reset_index(drop=True)

# 2) Leave-One-Out 划分：
#    每个用户：若交互<3条 → train+test；否则 → train+valid+test
def loo_split(df):
    train, valid, test = [], [], []
    for uid, grp in df.groupby("uid"):
        mids = grp.mid.values.tolist()
        if len(mids) < 3:
            train += [(uid, m) for m in mids[:-1]]
            test.append((uid, mids[-1]))
        else:
            train += [(uid, m) for m in mids[:-2]]
            valid.append((uid, mids[-2]))
            test.append((uid, mids[-1]))
    return (
        pd.DataFrame(train, columns=["uid","mid"]),
        pd.DataFrame(valid, columns=["uid","mid"]),
        pd.DataFrame(test,  columns=["uid","mid"]),
    )

train_df, valid_df, test_df = loo_split(ratings)

# 保存划分结果
train_df.to_csv(f"{CACHE}/ui_train.csv", index=False)
valid_df.to_csv(f"{CACHE}/ui_valid.csv", index=False)
test_df. to_csv(f"{CACHE}/ui_test.csv",  index=False)

print(f"▸ train/valid/test 交互数：{len(train_df)}/{len(valid_df)}/{len(test_df)}")


# 4) Movies：提取 Genre + Decade 特征
movies = pd.read_csv(
    f"{ROOT}/movies.dat",
    sep="::",
    engine="python",
    names=["mid","title","genres"],
    encoding="latin-1"
)

# 提取上映年份并映射到十年档
movies["year"]   = movies.title.str.extract(r"\((\d{4})\)").astype(float)
movies["decade"] = (movies.year // 10 * 10).fillna(0).astype(int)

# 构造 Genre one-hot
all_genres = [
    'Action','Adventure','Animation',"Children's",'Comedy','Crime',
    'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',
    'Mystery','Romance','Sci-Fi','Thriller','War','Western'
]
genre_map = {g:i for i,g in enumerate(all_genres)}
genre_vecs = np.zeros((len(movies), len(all_genres)), dtype=np.float32)
for idx, row in movies.iterrows():
    for g in row.genres.split('|'):
        if g in genre_map:
            genre_vecs[idx, genre_map[g]] = 1.0

# 构造 Decade one-hot
decades = sorted(movies.decade.unique())
dec_map = {d:i for i,d in enumerate(decades)}
dec_vecs = np.zeros((len(movies), len(decades)), dtype=np.float32)
for idx, row in movies.iterrows():
    dec_vecs[idx, dec_map[row.decade]] = 1.0

# 合并内容特征并保存
feat = np.concatenate([genre_vecs, dec_vecs], axis=1)
np.save(f"{CACHE}/item_feat.npy", feat)
movies[["mid"]].to_csv(f"{CACHE}/item_idx.csv", index=False)

print("✓ preprocess_basic done")
print("  item_feat shape:", feat.shape)
print("  num_items:", len(movies))


