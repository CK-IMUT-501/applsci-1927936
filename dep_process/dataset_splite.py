#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch
import torchtext

from sklearn.model_selection import train_test_split

import random
import re
# from tqdm import tqdm  # 进度条
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import unicodedata
import datetime
import time
import copy
import math

import sacrebleu
from nltk.translate.bleu_score import sentence_bleu
import pickle
import torch_optimizer as optim
# import adamod
import os
import shutil
import sys


# In[2]:


print('dataset_splite.ipynb')

MAX_LENGTH = int(sys.argv[1])  # MAX_LENGTH = 100
print('MAX_LENGTH：', MAX_LENGTH)
current_datasets_path = sys.argv[2] #'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/en_zh/'
print('current_datasets_path：', current_datasets_path)
parallel_corpus = sys.argv[3] #'news_en_zh_shuffle_final.txt'
print('parallel_corpus：', parallel_corpus)

train_pairs_path = current_datasets_path + 'train_pairs'
val_pairs_path = current_datasets_path + 'val_pairs'
test_pairs_path = current_datasets_path + 'test_pairs'


# In[3]:


# 数据读取
# 当你用read_csv读文件的时候，如果文本里包含英文双引号，直接读取会导致行数变少或是直接如下报错停止
# 此时应该对read_csv设置参数控制csv中的引号常量，设定quoting=3或是quoting=csv.QUOTE_NONE”（注：用第二种要先导csv库）然后问题就解决了。

data_df = pd.read_csv(current_datasets_path + parallel_corpus,  # 数据格式：英语\t法语，注意我们的任务源语言是法语，目标语言是英语
                      encoding='UTF-8', sep='\t', header=None,quoting=3,
                      names=['src', 'targ'], index_col=False)

# print(data_df.shape)
# print(data_df.values.shape)
# print(data_df.values[0])
# print(data_df.values[0].shape)
# data_df.head()


# In[4]:


# 数据预处理

# 规范化字符串
def normalizeString(s):
    # print(s) # list  ['Go.']
    # s = s[0]
    s = s.lower().strip()
    #s = unicodeToAscii(s)
    #s = re.sub(r"([.!?])", r" \1", s)  # \1表示group(1)即第一个匹配到的 即匹配到'.'或者'!'或者'?'后，一律替换成'空格.'或者'空格!'或者'空格？'
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # 非字母以及非.!?的其他任何字符 一律被替换成空格
    s = re.sub(r'[\s]+', " ", s)  # 将出现的多个空格，都使用一个空格代替。例如：w='abc  1   23  1' 处理后：w='abc 1 23 1'
    return s


# print(normalizeString('Va !'))
# print(normalizeString('Go.'))


# In[5]:


pairs = [[normalizeString(s) for s in line] for line in data_df.values]

print('pairs num=', len(pairs))
# print(pairs[0])
# print(pairs[0])


# In[6]:


# 文件是英译法，我们实现的是法译英，所以进行了reverse，所以pair[1]是英语
# 为了快速训练，仅保留“我是”“你是”“他是”等简单句子，并且删除原始文本长度大于10个标记的样本
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    # 过滤，并交换句子顺序，得到法英句子对（之前是英法句子对）
    return [[pair[0], pair[1]] for pair in pairs if filterPair(pair)]


pairs = filterPairs(pairs)

print('经过过滤后平行语料数目为：', len(pairs))
# print(pairs[0])
# print(random.choice(pairs))
# print(np.array(pairs).shape)


# In[7]:


# 划分数据集：训练集和验证集
##蒙汉50w 0.020 0.020
##英汉
train_test, val_pairs = train_test_split(pairs, test_size=0.030, random_state=1234)
train_pairs, test_pairs = train_test_split(train_test, test_size=0.030, random_state=1234)

# print('训练集句子数目：', len(train_pairs))
# print('验证集句子数目：', len(val_pairs))
# print('测试集句子数目：', len(test_pairs))
# print(test_pairs[0])


# In[8]:


with open(train_pairs_path,'wb') as f:
    pickle.dump(train_pairs, f)
    f.close()

with open(val_pairs_path,'wb') as f:
    pickle.dump(val_pairs, f)
    f.close()

with open(test_pairs_path,'wb') as f:
    pickle.dump(test_pairs, f)
    f.close()


# In[9]:


print('训练集句子数目：', len(train_pairs))
print('验证集句子数目：', len(val_pairs))
print('测试集句子数目：', len(test_pairs))
print('运行结束')


# In[10]:


# with open(r'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/en_zh/train_pairs','rb') as f:
#     train_pairs = pickle.load(f)
#     f.close()

# with open(r'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/en_zh/val_pairs','rb') as f:
#     val_pairs = pickle.load(f)
#     f.close()

# with open(r'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/en_zh/test_pairs','rb') as f:
#     test_pairs = pickle.load(f)
#     f.close()

# print('训练集句子数目：', len(train_pairs))
# print('验证集句子数目：', len(val_pairs))
# print('测试集句子数目：', len(test_pairs))

