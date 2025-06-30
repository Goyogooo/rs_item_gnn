import pandas as pd
import numpy as np

# 假设数据集已经加载为 DataFrame `ratings`
ratings = pd.read_csv(
    "data/ml-1m/ratings.dat",  # 这里是数据路径，调整为实际路径
    sep="::",
    engine="python",
    names=["uid", "mid", "rate", "ts"],
    encoding="latin-1"
)

# 1. 统计总用户数、总电影数、总交互数
num_users = ratings['uid'].nunique()  # 统计唯一用户数
num_movies = ratings['mid'].nunique()  # 统计唯一电影数
num_interactions = len(ratings)  # 统计总交互数

# 2. 用户交互小于3的用户数和交互大于等于3的用户数
user_interactions_count = ratings.groupby('uid').size()  # 按用户分组并统计每个用户的交互数量
users_less_than_3 = user_interactions_count[user_interactions_count < 3].count()  # 交互少于3的用户数
users_3_or_more = user_interactions_count[user_interactions_count >= 3].count()  # 交互大于等于3的用户数

# 打印统计结果
print(f"总用户数: {num_users}")
print(f"总电影数: {num_movies}")
print(f"总交互数据数: {num_interactions}")
print(f"交互小于3的用户数: {users_less_than_3}")
print(f"交互大于等于3的用户数: {users_3_or_more}")

# 设置数据路径
ROOT = "data/ml-1m"
CACHE = f"{ROOT}/cache"

# 加载 Movies 数据
movies = pd.read_csv(
    f"{ROOT}/movies.dat",
    sep="::",
    engine="python",
    names=["mid", "title", "genres"],
    encoding="latin-1"
)

# 加载划分后的数据集（训练集、验证集、测试集）
train_df = pd.read_csv(f"{CACHE}/ui_train.csv")
valid_df = pd.read_csv(f"{CACHE}/ui_valid.csv")
test_df = pd.read_csv(f"{CACHE}/ui_test.csv")

# 处理 Decade 映射
movies["year"] = movies.title.str.extract(r"\((\d{4})\)").astype(float)
movies["decade"] = (movies.year // 10 * 10).fillna(0).astype(int)

# 获取 Decade 的范围
decades = sorted(movies.decade.unique())
print(f"Decade 映射编码范围: {min(decades)} 到 {max(decades)}")

# 输出训练集、验证集和测试集的交互数
print(f"▸ train/valid/test 交互数：{len(train_df)}/{len(valid_df)}/{len(test_df)}")

# 设置合并后的特征文件路径
item_feat_path = f"{CACHE}/item_feat.npy"

# 加载合并后的特征文件
item_feat = np.load(item_feat_path)

# 输出合并后特征的维度
print(f"合并后的电影特征维度: {item_feat.shape}")