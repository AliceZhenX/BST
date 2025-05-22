#The code is modified based on https://github.com/F-Yousefi/RecSys-BST-Pytorch.git

from pathlib import Path
import pandas as pd
from torch.utils import data
import torch

def read_dat(path, columns):
  try:
    df = pd.read_csv(
      path,
      sep="::",
      names=columns,
      engine='python',
      )
  except:
    df = pd.read_csv(
      path,
      sep="::",
      names=columns,
      engine='python',
      encoding="ISO-8859-1"
      )
  return df

file_name = Path("ml-1m")
files_list = ["movies.dat","ratings.dat","users.dat"] #文件名
files_colums = [["movie_id", "title", "genres"],#movie中的列名
 ["user_id", "movie_id", "rating", "timestamps"], #rating中的列名
  ["user_id", "gender", "age_group", "occupation", "zip_code"]] #user中的列名

movies_org, ratings_org, users_org = \
 [read_dat(file_name / files_list[i], columns = files_colums[i]) for i in range(len(files_list))]

movies, ratings, users = movies_org.copy(), ratings_org.copy(), users_org.copy()

def convert_to_codes(df, columns):
    for col in columns:
        df[col] = df[col].astype('category').cat.codes

convert_to_codes(ratings, ["movie_id", "timestamps"])
convert_to_codes(movies, ["movie_id"])
convert_to_codes(users, users.columns.tolist())

users_ratings = pd.merge(left=ratings, right= users, on="user_id")
users_ratings = users_ratings.drop(columns = ['zip_code']) #[1000209 rows x 7 columns]

#处理评分数据，进行最大最小归一化
users_ratings.rating = (users_ratings.rating - users_ratings.rating.min()) / \
                      (users_ratings.rating.max() - users_ratings.rating.min())

user_groups = users_ratings.sort_values(by=["timestamps"]).groupby("user_id")
dataset = pd.DataFrame(data = {
        "user_id": list(user_groups.groups.keys()),
        "gender" : list(user_groups.gender.unique().explode()),
        "age_group" : list(user_groups.age_group.unique().explode()),
        "occupation" : list(user_groups.occupation.unique().explode()),#单个值
        "movie_ids": list(user_groups.movie_id.apply(list)),
        "ratings": list(user_groups.rating.apply(list)),
        "timestamps": list(user_groups.timestamps.apply(list)),#apply(list)列表的形式"timestamps": [964982703, 964983030]
    })

#处理电影数据
genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

movies[genres] = movies["genres"].str.get_dummies(sep='|')
USER_FEATUERS = ["user_id","gender", "age_group", "occupation"]
FEATURES_VOCABULARY = {
    "user_id" : users.user_id.tolist(),
    "gender" : users.gender.unique().tolist(),
    "occupation": users.occupation.unique().tolist(),
    "age_group": users.age_group.unique().tolist(),
    "movie_id": movies.movie_id.tolist(),
    "genres": movies[genres].to_numpy()
}
PARAMETERS ={
    "sequence_length": 4,
    "step": 1,
}
def create_sequences(values, seq_len, step):#用滑动窗口
    return [values[i:i+seq_len]
           for i in range(0, len(values)-seq_len+1, step)]
for col in ["movie_ids", "ratings"]:
    dataset[col] = dataset[col].apply(
        lambda x: create_sequences(x, PARAMETERS["sequence_length"], PARAMETERS["step"])
    )

dataset = dataset.drop(columns = ["timestamps"])
dataset = dataset.explode(column=["ratings", "movie_ids"]).reset_index(drop=True)

class MovieDataset(data.Dataset):
  def __init__(self, dataset):
    self.len = len(dataset)
    self.user_id = torch.tensor(dataset.user_id.values, dtype=int)
    self.gender = torch.tensor(dataset.gender.values, dtype=int)
    self.occupation = torch.tensor(dataset.occupation.values, dtype=int)
    self.age_group = torch.tensor(dataset.age_group.values, dtype=int)
    self.movie_ids = torch.tensor(dataset.movie_ids.tolist(), dtype=int)
    self.ratings = torch.tensor(dataset.ratings.tolist(), dtype=torch.float32)

  def __len__(self):
    return self.len


  def __getitem__(self, idx):
    if isinstance(idx, slice):
      raise 0
    sequence_movie_ids = self.movie_ids[idx][:-1]
    target_movie_id = self.movie_ids[idx][-1:]
    sequence_ratings = self.ratings[idx][:-1]
    target_rating = self.ratings[idx][-1:]


    return self.user_id[idx], self.gender[idx], self.age_group[idx],\
           self.occupation[idx], sequence_movie_ids,\
           target_movie_id, sequence_ratings, target_rating

from sklearn.model_selection import train_test_split
import numpy as np

# 重新划分数据
train_data, test_data = train_test_split(
    dataset,
    test_size=0.2,  # 拿出 20% 的数据作为测试集
    stratify=dataset[['gender', 'age_group']],
    random_state=42
)

train_data = MovieDataset(train_data)
test_data = MovieDataset(test_data)

# 定义 NDCG 计算函数
def dcg_score(y_true, y_score, k=5):
    """计算折损累积增益（DCG）"""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=5):
    """计算归一化折损累积增益（NDCG）"""
    best_dcg = dcg_score(y_true, y_true, k)
    if best_dcg == 0:
        return 0
    return dcg_score(y_true, y_score, k) / best_dcg

import lightning as ltorch
from torch import nn
import torch
import math

class MovieLens(ltorch.LightningModule):

  def __init__(self):
    super().__init__()
    len_movie_voc = len(FEATURES_VOCABULARY["movie_id"])
    self.len_movie_embedded = int(math.sqrt(len_movie_voc))+2 #保证可以被头整除 此时为64
    dropout_rate = 0.15
    num_heads = 4

    for idx,feature in enumerate(USER_FEATUERS):
      num_embeddings=len(FEATURES_VOCABULARY[feature])
      embedding_dim= int(math.sqrt(len(FEATURES_VOCABULARY[feature])))
      emb = nn.Embedding(
          num_embeddings=num_embeddings,
          embedding_dim= embedding_dim
          )
      setattr(self, f'emb_{idx}', emb)

    self.sequence_movie_embedding = nn.Embedding(
                                        num_embeddings=len_movie_voc,
                                        embedding_dim=self.len_movie_embedded)

    self.genres_embedding = nn.Embedding(
                        num_embeddings=FEATURES_VOCABULARY["genres"].shape[0],#所有数据
                        embedding_dim= FEATURES_VOCABULARY["genres"].shape[-1]) #每条数据对应的01表示的种类,18
    self.genres_embedding.weight = torch.nn.Parameter( #all weights are initialized ...
                torch.from_numpy(FEATURES_VOCABULARY["genres"].astype(float))) #为什么这里不需要训练？这里直接是独热编码
    self.genres_embedding.weight.requires_grad=False #trainable = False


    in_features_num = FEATURES_VOCABULARY["genres"].shape[-1] + self.len_movie_embedded #种类，电影id嵌入长度

    self.movies_sequence_dense =nn.Sequential(
            nn.Linear(in_features=in_features_num * (PARAMETERS["sequence_length"] -1),
                      out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,
                      out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,
                      out_features=(PARAMETERS["sequence_length"] -1) * self.len_movie_embedded),
            nn.BatchNorm1d(num_features=(PARAMETERS["sequence_length"] -1) * self.len_movie_embedded)
    )#对应论文中itemseq数据

    position = torch.arange(PARAMETERS["sequence_length"] -1).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0,self.len_movie_embedded, 2) * (-math.log(10000.0) / 60))

    self.positional_embedding = torch.zeros(
        PARAMETERS["sequence_length"] -1,#减1排除最后一个位置
        self.len_movie_embedded)

    self.positional_embedding[:, 0::2] = torch.sin(position * div_term)
    self.positional_embedding[:, 1::2] = torch.cos(position * div_term)

    if torch.cuda.is_available():
        self.positional_embedding = self.positional_embedding.to("cuda")
    # self.register_buffer("positional_embedding", self.positional_embedding)
    # self.register_buffer(
    #     "positional_embedding",
    #     torch.zeros(PARAMETERS["sequence_length"] - 1, self.len_movie_embedded)
    # )

    self.target_movie_embedding = nn.Embedding(
                                      num_embeddings=len_movie_voc,
                                      embedding_dim= self.len_movie_embedded)

    self.movies_target_dense = nn.Sequential(
            nn.Linear(in_features=in_features_num,
                                      out_features=256),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Linear(in_features=256,
                                      out_features=128),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Linear(in_features=128,
                                      out_features=self.len_movie_embedded),
            nn.BatchNorm1d(num_features=1)##????相当于保留时间步，计算所有样本和时间步上的均值于方差
    )

    self.multi_head =nn.MultiheadAttention(
                            embed_dim= self.len_movie_embedded,
                            num_heads= num_heads,
                            dropout=dropout_rate)

    self.dropout_first = nn.Dropout(dropout_rate)

    self.norm_transformer_first = nn.BatchNorm1d(PARAMETERS["sequence_length"])

    self.dense_transformer = nn.Linear(in_features=self.len_movie_embedded * PARAMETERS["sequence_length"],
                             out_features=self.len_movie_embedded * PARAMETERS["sequence_length"])

    self.norm_transformer_second = nn.BatchNorm1d(PARAMETERS["sequence_length"])

    self.dropout_second = nn.Dropout(dropout_rate)

    self.fully_connected = nn.Sequential(
        nn.Linear(in_features=84 + (PARAMETERS["sequence_length"] * self.len_movie_embedded),
                  out_features=256),
        nn.BatchNorm1d(num_features=256),
        nn.LeakyReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(in_features=256,out_features=128),
        nn.BatchNorm1d(num_features=128),
        nn.LeakyReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(in_features=128,out_features=1), #层数不一样，激活函数呢？
        )
    self.loss_function = nn.functional.mse_loss

  def other_features_encoder(self, batch):
    concat_other_features = []
    for idx,input in enumerate(batch):
      concat_other_features.append(getattr(self, f'emb_{idx}')(input))
    #根据变量 idx 的值，自动选择一个嵌入层（如 self.emb_0, self.emb_1 等），然后用这个嵌入层对输入 input 进行前向传播。
    other_features = torch.cat(concat_other_features, dim=-1)
    return other_features

  def seq_mov_emb(self, batch):
    sme = self.sequence_movie_embedding(batch)
    ge = self.genres_embedding(batch)
    cat_sme = torch.cat([sme,ge],dim = -1).float()
    out = self.dense_trainer(
         self.movies_sequence_dense,
         cat_sme,
         (batch.shape[0],PARAMETERS["sequence_length"] -1, self.len_movie_embedded))

    return nn.functional.relu(out)

  def tar_mov_emb(self, batch):
    tme = self.target_movie_embedding(batch)
    ge = self.genres_embedding(batch)
    cat_sme = torch.cat([tme,ge],dim = -1).float()
    out = self.movies_target_dense(cat_sme)
    return nn.functional.relu(out)

  def transformer(self, ts):
    mh, _ = self.multi_head(ts,ts,ts)
    drmh = self.dropout_first(mh)
    add = ts + drmh
    norm_f = self.norm_transformer_first(add)
    lcr = nn.functional.leaky_relu(norm_f)
    dst = self.dense_trainer(self.dense_transformer,lcr)
    drds = self.dropout_second(dst)
    add_nd = norm_f + drds
    norm_s = self.norm_transformer_second(add_nd)
    flat = torch.flatten(norm_s, start_dim=1)#第一维为batch
    return flat

  def fully_connected_model(self, input):
    return self.fully_connected(input)

  def dense_trainer(self, layer, input, output_shape= None):
    input_shape = input.shape
    input = torch.flatten(input, start_dim=1)
    output = layer(input)

    if output_shape == None : output = torch.reshape(output, input_shape)
    else: output = torch.reshape(output, output_shape)

    return output

  def training_step(self, batch, batch_idx=0):
    self.train()
    other_features = self.other_features_encoder(batch[:4])
    sme = self.seq_mov_emb(batch[4])
    tme = self.tar_mov_emb(batch[5])
    sme = self.positional_embedding + sme
    sr = batch[6].unsqueeze(dim = -1)
    mul_tme_sr = torch.mul(sr, sme)
    transformer_features = torch.cat([tme,mul_tme_sr], dim=-2)
    flat = self.transformer(transformer_features)
    cat_tr_oth = torch.cat([other_features,flat],dim = -1)
    y_pred = self.fully_connected_model(cat_tr_oth) #没有用sigmoid层
    y = batch[7]
    loss = self.loss_function(y_pred, y)
    mae = nn.functional.l1_loss(y_pred, y) #电影评分预测是回归问题
    mse = nn.functional.mse_loss(y_pred, y)
    self.log_dict({"Train Loss": loss, "Train l1 loss":mae,"Train mse loss":mse  },on_epoch=True, prog_bar=True, enable_graph=True )
    return loss

  def validation_step(self, batch, batch_idx=0):
    self.eval()
    other_features = self.other_features_encoder(batch[:4]) #不同层的嵌入，user_id,性别，年龄，职业
    sme = self.seq_mov_emb(batch[4]) #电影嵌入，id和类型
    tme = self.tar_mov_emb(batch[5]) #目标电影嵌入，id和类型
    sme = self.positional_embedding + sme #位置编码
    sr = batch[6].unsqueeze(dim = -1) #序列电影的打分
    mul_tme_sr = torch.mul(sr, sme)
    transformer_features = torch.cat([tme,mul_tme_sr], dim=-2)
    flat = self.transformer(transformer_features)
    cat_tr_oth = torch.cat([other_features,flat],dim = -1)
    y_pred = self.fully_connected_model(cat_tr_oth)
    y = batch[7]
    loss = self.loss_function(y_pred, y)
    self.log_dict({"Validation Loss": loss},on_epoch=True, prog_bar=True,on_step = False, enable_graph=True)

  def predict_step(self, batch, batch_idx=0):
    self.eval()
    other_features = self.other_features_encoder(batch[:4])
    sme = self.seq_mov_emb(batch[4])
    tme = self.tar_mov_emb(batch[5])
    sme = self.positional_embedding + sme
    sr = batch[6].unsqueeze(dim = -1)
    mul_tme_sr = torch.mul(sr, sme)
    transformer_features = torch.cat([tme,mul_tme_sr], dim= 1)

    flat = self.transformer(transformer_features)
    cat_tr_oth = torch.cat([other_features,flat],dim = -1)
    y_pred = self.fully_connected_model(cat_tr_oth)
    return torch.cat([batch[4], batch[5], y_pred], dim=1)

  def test_step(self, batch, batch_idx=0):
      self.eval()
      other_features = self.other_features_encoder(batch[:4])
      sme = self.seq_mov_emb(batch[4])
      tme = self.tar_mov_emb(batch[5])
      sme = self.positional_embedding + sme
      sr = batch[6].unsqueeze(dim=-1)
      mul_tme_sr = torch.mul(sr, sme)
      transformer_features = torch.cat([tme, mul_tme_sr], dim=-2)
      flat = self.transformer(transformer_features)
      cat_tr_oth = torch.cat([other_features, flat], dim=-1)
      y_pred = self.fully_connected_model(cat_tr_oth)
      y = batch[7]

      # 计算 NDCG
      y_true = y.cpu().numpy().flatten()
      y_score = y_pred.cpu().numpy().flatten()
      ndcg = ndcg_score(y_true, y_score, k=5)

      self.log_dict({"Test NDCG": ndcg}, on_epoch=True, prog_bar=True, on_step=False, enable_graph=True)

  def configure_optimizers(self):
    optimizer = torch.optim.Adagrad(self.parameters(), lr=1e-3, weight_decay=5e-6)
    return optimizer


model = MovieLens()

from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# create your own theme!
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="#8756d6",
        metrics_text_delimiter="\n",
        metrics_format=".3f",
    )
)
trainer = ltorch.Trainer(max_epochs=100, accelerator="auto",
                         devices="auto", strategy="auto",
                         callbacks=[EarlyStopping(
                             monitor="Validation Loss", mode="min"),
                                    progress_bar])
trainer.fit(model=model,
            train_dataloaders=torch.utils.data.DataLoader(
                train_data, batch_size=1024, shuffle=True, num_workers=0),
            val_dataloaders=torch.utils.data.DataLoader(
                test_data, batch_size=1024, shuffle=False, num_workers=0))

trainer.test(model, dataloaders=torch.utils.data.DataLoader(
    test_data, batch_size=1024, shuffle=False, num_workers=0))




