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

path = ""
model = SentenceTransformer('')


def get_datas_embedding(datas):
    return model.encode(datas)


datas = ["我喜欢小丁的文章", "我讨厌小丁的创作内容", "我非常喜欢小丁写的文章"]
datas_embedding = get_datas_embedding(datas)
