#!/usr/bin/env python
# coding: utf-8

# In[18]:


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


# In[17]:


# 设置超参数
print(' ')
print('dep_save.ipynb')

ngpu = int(sys.argv[1]) #2
print('ngpu：', ngpu)
batch = int(sys.argv[2]) #120
print('batch：', batch)
# MAX_LENGTH = d_model//num_heads
MAX_LENGTH = int(sys.argv[3]) #100
print('MAX_LENGTH：', MAX_LENGTH)
current_datasets_path = sys.argv[4] #'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/en_zh/'
print('current_datasets_path：', current_datasets_path)

train_pairs_path = current_datasets_path + 'train_pairs'
val_pairs_path = current_datasets_path + 'val_pairs'
test_pairs_path = current_datasets_path + 'test_pairs'


# In[3]:


# # 数据读取
# # 当你用read_csv读文件的时候，如果文本里包含英文双引号，直接读取会导致行数变少或是直接如下报错停止
# # 此时应该对read_csv设置参数控制csv中的引号常量，设定quoting=3或是quoting=csv.QUOTE_NONE”（注：用第二种要先导csv库）然后问题就解决了。

# data_dir = '/home/chengkun/jupyter_projects/Magic-NLPer-main/data/' 

# data_df = pd.read_csv(data_dir + 'ch_mn_50_nodict.txt',  # 数据格式：英语\t法语，注意我们的任务源语言是法语，目标语言是英语
#                       encoding='UTF-8', sep='\t', header=None,quoting=3,
#                       names=['src', 'targ'], index_col=False)

# # print(data_df.shape)
# # print(data_df.values.shape)
# # print(data_df.values[0])
# # print(data_df.values[0].shape)
# # data_df.head()


# In[4]:


# # 数据预处理

# # 规范化字符串
# def normalizeString(s):
#     # print(s) # list  ['Go.']
#     # s = s[0]
#     s = s.lower().strip()
#     #s = unicodeToAscii(s)
#     #s = re.sub(r"([.!?])", r" \1", s)  # \1表示group(1)即第一个匹配到的 即匹配到'.'或者'!'或者'?'后，一律替换成'空格.'或者'空格!'或者'空格？'
#     #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # 非字母以及非.!?的其他任何字符 一律被替换成空格
#     s = re.sub(r'[\s]+', " ", s)  # 将出现的多个空格，都使用一个空格代替。例如：w='abc  1   23  1' 处理后：w='abc 1 23 1'
#     return s


# # print(normalizeString('Va !'))
# # print(normalizeString('Go.'))


# In[5]:


# pairs = [[normalizeString(s) for s in line] for line in data_df.values]

# print('pairs num=', len(pairs))
# # print(pairs[0])
# # print(pairs[0])


# In[6]:


# # 文件是英译法，我们实现的是法译英，所以进行了reverse，所以pair[1]是英语
# # 为了快速训练，仅保留“我是”“你是”“他是”等简单句子，并且删除原始文本长度大于10个标记的样本
# def filterPair(p):
#     return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH 

# def filterPairs(pairs):
#     # 过滤，并交换句子顺序，得到法英句子对（之前是英法句子对）
#     return [[pair[1], pair[0]] for pair in pairs if filterPair(pair)]


# pairs = filterPairs(pairs)

# print('经过过滤后平行语料数目为：', len(pairs))
# # print(pairs[0])
# # print(random.choice(pairs))
# # print(np.array(pairs).shape)


# In[7]:


# # 划分数据集：训练集和验证集
# ##0.0338 0.03485
# ##50 0.020 0.020
# train_test, val_pairs = train_test_split(pairs, test_size=0.020, random_state=1234)
# train_pairs, test_pairs = train_test_split(train_test, test_size=0.020, random_state=1234)

# print('训练集句子数目：', len(train_pairs))
# print('验证集句子数目：', len(val_pairs))
# print('测试集句子数目：', len(test_pairs))
# # print(test_pairs[0])


# In[8]:


# with open(r'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/train_pairs','wb') as f:
#     pickle.dump(train_pairs, f)
#     f.close()

# with open(r'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/val_pairs','wb') as f:
#     pickle.dump(val_pairs, f)
#     f.close()

# with open(r'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/test_pairs','wb') as f:
#     pickle.dump(test_pairs, f)
#     f.close()


# In[9]:


with open(train_pairs_path,'rb') as f:
    train_pairs = pickle.load(f)
    f.close()

with open(val_pairs_path,'rb') as f:
    val_pairs = pickle.load(f)
    f.close()

with open(test_pairs_path,'rb') as f:
    test_pairs = pickle.load(f)
    f.close()

print('训练集句子数目：', len(train_pairs))
print('验证集句子数目：', len(val_pairs))
print('测试集句子数目：', len(test_pairs))


# In[10]:


tokenizer = lambda x: x.split() # 分词器

SRC_TEXT = torchtext.data.Field(sequential=True,
                                tokenize=tokenizer,
                                # lower=True,
                                fix_length=MAX_LENGTH + 2,
                                preprocessing=lambda x: ['<start>'] + x + ['<end>'],
                                # after tokenizing but before numericalizing
                                # postprocessing # after numericalizing but before the numbers are turned into a Tensor
                                )
TARG_TEXT = torchtext.data.Field(sequential=True,
                                 tokenize=tokenizer,
                                 # lower=True,
                                 fix_length=MAX_LENGTH + 2,
                                 preprocessing=lambda x: ['<start>'] + x + ['<end>'],
                                 )


def get_dataset(pairs, src, targ):
    fields = [('src', src), ('targ', targ)]  # filed信息 fields dict[str, Field])
    examples = []  # list(Example)
    for src, targ in pairs: # 进度条
        # 创建Example时会调用field.preprocess方法
        examples.append(torchtext.data.Example.fromlist([src, targ], fields))
    return examples, fields


# examples, fields = get_dataset(pairs, SRC_TEXT, TARG_TEXT)

ds_train = torchtext.data.Dataset(*get_dataset(train_pairs, SRC_TEXT, TARG_TEXT))
ds_val = torchtext.data.Dataset(*get_dataset(val_pairs, SRC_TEXT, TARG_TEXT))
ds_test = torchtext.data.Dataset(*get_dataset(test_pairs, SRC_TEXT, TARG_TEXT))


# In[11]:


# # 查看1个样本的信息
print('ds_train')
print(len(ds_train[0].src), ds_train[0].src)
print(len(ds_train[0].targ), ds_train[0].targ)
# print('ds_val')
# print(len(ds_val[0].src), ds_val[0].src)
# print(len(ds_val[0].targ), ds_val[0].targ)
# print('ds_test')
# print(len(ds_test[0].src), ds_test[0].src)
# print(len(ds_test[0].targ), ds_test[0].targ)


# In[12]:


# 构建词典
print('模型大小与词表大小正相关，控制词表大小')
SRC_TEXT.build_vocab(ds_train,min_freq=1)  # 建立词表 并建立token和ID的映射关系
# print(len(SRC_TEXT.vocab))
# print(SRC_TEXT.vocab.itos[0])
# print(SRC_TEXT.vocab.itos[1])
# print(SRC_TEXT.vocab.itos[2])
# print(SRC_TEXT.vocab.itos[3])
# print(SRC_TEXT.vocab.stoi['<start>'])
# print(SRC_TEXT.vocab.stoi['<end>'])

# 模拟decode
res = []
for id in range(20):
    res.append(SRC_TEXT.vocab.itos[id])
print('0-20：'+' '.join(res)+'\n')

TARG_TEXT.build_vocab(ds_train,min_freq=1)

# print(len(TARG_TEXT.vocab))
# print(TARG_TEXT.vocab.itos[0])
# print(TARG_TEXT.vocab.itos[1])
# print(TARG_TEXT.vocab.itos[2])
# print(TARG_TEXT.vocab.itos[3])
# print(TARG_TEXT.vocab.stoi['<start>'])
# print(TARG_TEXT.vocab.stoi['<end>'])

input_vocab_size = len(SRC_TEXT.vocab)
target_vocab_size = len(TARG_TEXT.vocab)

print('input_vocab_size：', input_vocab_size)
print('target_vocab_size：', target_vocab_size)


# In[13]:


BATCH_SIZE = batch * ngpu

# 构建数据管道迭代器
train_iter, val_iter, test_iter= torchtext.data.Iterator.splits(
    (ds_train, ds_val, ds_test),
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    batch_sizes=(BATCH_SIZE, BATCH_SIZE, BATCH_SIZE)
)

# train_iter = torchtext.data.Iterator.splits(
#     (ds_train),
#     sort_within_batch=True,
#     sort_key=lambda x: len(x.src),
#     batch_sizes=(BATCH_SIZE)
# )

# # 查看数据管道信息，此时会触发postprocessing，如果有的话
# for BATCH in train_iter:
#     # 注意，这里text第0维不是batch，而是seq_len
#     print(BATCH.src.shape, BATCH.targ.shape)  # [12,64], [12,64]
#     break


# In[14]:


# 将数据管道组织成与torch.utils.data.DataLoader相似的inputs, targets的输出形式
class DataLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)  # 一共有多少个batch？

    def __len__(self):
        return self.length

    def __iter__(self):
        # 注意，在此处调整text的shape为batch first
        for batch in self.data_iter:
            yield (torch.transpose(batch.src, 0, 1), torch.transpose(batch.targ, 0, 1))


train_dataloader = DataLoader(train_iter)
val_dataloader = DataLoader(val_iter)
test_dataloader = DataLoader(test_iter)


# In[15]:


# 查看数据管道
print('len(train_dataloader):', len(train_dataloader))  # 句子总数/batch数
print('len(val_dataloader):', len(val_dataloader))  # 句子总数/batch数
print('len(test_dataloader):', len(test_dataloader))  # 句子总数/batch数

# for batch_src, batch_targ in train_dataloader:
#     print('batch_src.shape:',batch_src.shape,'\n','batch_targ.shape:',batch_targ.shape)  # [256,12], [256,12]
#     print(batch_src, batch_src.dtype)
#     print(batch_targ, batch_targ.dtype)
#     break


# In[16]:


def dep_save(dataloader, dep_file_path, dep_batch_file_path):
    
    if os.path.exists(dep_batch_file_path):  # 如果文件存在则删除文件，可使用以下两种方法。
        os.remove(dep_batch_file_path) 
        print('已删除旧文件dep_train_batch_pairs')
        #os.unlink(dep_batch_file)

    dep_batch_file = open(dep_batch_file_path,'wb')
    
    for step, (inp, targ) in enumerate(dataloader, start=1):
        count = 0
        dependency_matrix = torch.zeros(1,inp.shape[1]-1,inp.shape[1]-1)
        while count < inp.shape[0]:
            try:
                dependency_matrix = torch.cat((dependency_matrix,pickle.load(dep_file_path)),0)
                count = count + 1
            except EOFError:
                print('处理完毕')
                break
        
        dependency_matrix = dependency_matrix[1:,:,:]
        pickle.dump(dependency_matrix,dep_batch_file)
        
        if step%1 == 0:
            print(step, dependency_matrix.shape)
            
    dep_batch_file.close()
    dep_file_path.close()
    return True


# In[ ]:


# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(' ','_'))
# dep_file_path = open(r'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/dep_val_pairs','rb')
# dep_batch_file_path = r'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/dep_val_batch_pairs'
# dep_save(val_dataloader, dep_file_path, dep_batch_file_path)
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(' ','_'))


# In[19]:


print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(' ','_'))
dep_file_path = open(current_datasets_path + 'dep_train_pairs','rb')
dep_batch_file_path = current_datasets_path + 'dep_train_batch_pairs'
dep_save(train_dataloader, dep_file_path, dep_batch_file_path)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(' ','_'))
print('运行结束')


# In[23]:


# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(' ','_'))
# dep_batch_file_path = r'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/dep_train_batch_pairs'
# f = open(dep_batch_file_path,'rb')
# for step, (inp, targ) in enumerate(train_dataloader, start=1):
    
# #     st = (step-1) * inp.shape[0]
# #     end = st + inp.shape[0]
#     dependency_matrix = pickle.load(f)
#     if step%1 == 0:
#         print(step,inp.shape,targ.shape,dependency_matrix.shape)
# f.close()
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(' ','_'))

