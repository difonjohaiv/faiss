#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :faiss_new.py
@说明        :入门级小白篇代码教程
@时间        :2023/10/05 21:16:09
@作者        :张三
@版本        :1.0
@引用        :https://zhuanlan.zhihu.com/p/642959732
'''

from sentence_transformers import SentenceTransformer
import faiss

path = "sbert-base-chinese-nli"
model = SentenceTransformer(path)


# 文本向量化
def get_datas_embedding(datas):
    return model.encode(datas)


# 创建索引
def create_index(datas_embedding):
    # 构建索引，这里我们选用暴力检索的方法FlatL2为例，L2代表构建的index采用的相似度度量方法为L2范数，即欧氏距离
    index = faiss.IndexFlatL2(datas_embedding.shape[1])  # 这里必须传入一个向量的维度，创建一个空的索引
    index.add(datas_embedding)  # 把向量数据加入索引
    return index


# 数据检索
def data_recall(faiss_index, query, top_k):
    query_embedding = model.encode([query])
    Distance, Index = faiss_index.search(query_embedding, top_k)
    return Index


# faiss索引保存
def faiss_index_save(faiss_index, save_file_location):
    faiss.write_index(faiss_index, save_file_location)


# 索引的加载
def faiss_index_load(faiss_index_save_file_location):
    index = faiss.read_index(faiss_index_save_file_location)
    return index


# 向索引中添加向量
def index_data_add(faiss_index):
    # 获得索引向量的数量
    print(faiss_index.ntotal)
    data = ["小丁的文章太好看了"]
    datas_embedding = get_datas_embedding(data)
    faiss_index.add(datas_embedding)
    print(faiss_index.ntotal)


# 索引中向量的删除
def index_data_delete(faiss_index):
    print(faiss_index.ntotal)
    # remove, 指定要删除的向量id，是一个np的array
    faiss_index.remove_ids()  # 删除id为0,1,2,3,4的向量
    # index.remove_ids(np.array([0]))
    print(faiss_index.ntotal)


datas = ["我喜欢小丁的文章", "我讨厌小丁的创作内容", "我非常喜欢小丁写的文章"]

datas_embedding = get_datas_embedding(datas)
faiss_index = create_index(datas_embedding=datas_embedding)

sim_data_Index = data_recall(faiss_index=faiss_index, query="我爱看小丁的文章", top_k=2)

print("相似的top2数据是：")
for index in sim_data_Index[0]:
    print(datas[int(index)] + "\n")
