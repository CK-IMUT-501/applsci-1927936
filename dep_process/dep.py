#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
#encoding=utf-8 

# nlp环境
# 测试stanfordcorenlp是否可用
# ('xx',父节点,子节点)
# token节点编号从1开始


import numpy as np
import torch
import pickle
import datetime
import shutil
import os
from stanfordcorenlp import StanfordCoreNLP
import sys
np.set_printoptions(threshold=np.inf)


# In[2]:


print(' ')
print('dep.ipynb')

MAX_LENGTH = int(sys.argv[1])
print('MAX_LENGTH:',MAX_LENGTH)
current_datasets_path = sys.argv[2] #'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/en_zh/'
print('current_datasets_path:',current_datasets_path)

train_pairs_path = current_datasets_path + 'train_pairs'
# val_pairs_path = current_datasets_path + 'val_pairs'
# test_pairs_path = current_datasets_path + 'test_pairs'


# In[3]:


with open(train_pairs_path,'rb') as f:
    train_pairs = pickle.load(f)
    f.close()
    
# with open(r'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/en_zh/val_pairs','rb') as f:
#     val_pairs = pickle.load(f)
#     f.close()
    
# with open(r'/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/en_zh/test_pairs','rb') as f:
#     test_pairs = pickle.load(f)
#     f.close()
    


# In[4]:


print(len(train_pairs),train_pairs[0][1],'\n',train_pairs[-1][1])
# print(len(val_pairs),val_pairs[0][1],'\n',val_pairs[-1][1])
# print(len(test_pairs),test_pairs[0][1],'\n',test_pairs[-1][1])


# In[14]:



def f1(pairs,save_file):
    nlp = StanfordCoreNLP(r'/home/chengkun/java/stanford-corenlp-4.2.2/', lang='zh', memory='8g')
    for step, item in enumerate(pairs):
        if step%5000==0:
            print(step, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        item = item[1].replace(' ','')
        length_item = len(item)
#         print(length_item, item)
        
        tokens =  nlp.word_tokenize(item)
        dep_outputs = nlp.dependency_parse(item) # 解析语法
        length_tokens = len(''.join(tokens))
#         print(length_tokens, tokens)
#         print()
        
    #     print(tokens)
    #     print(dep_outputs)
    # nlp.close()

        # 查找根结点对应的索引
        root_index=[]
        for i in range(len(dep_outputs)):
            if dep_outputs[i][0]=='ROOT':
                root_index.append(i)

        # 修改依存关系三元组
        new_dep_outputs=[]
        for i in range(len(dep_outputs)):
            for index in root_index:
                if i+1>index:
                    tag=index

            if dep_outputs[i][0]=='ROOT':
                dep_output=(dep_outputs[i][0],dep_outputs[i][1],dep_outputs[i][2]+tag)
            else:
                dep_output = (dep_outputs[i][0], dep_outputs[i][1] + tag, dep_outputs[i][2] + tag)
            new_dep_outputs.append(dep_output)

    #     print('new_dep_outputs','\n',new_dep_outputs)

        jilu = []
        count = -1
        for step1,token in enumerate(tokens):
            t = [step1]
            for step2,char in enumerate(token):
    #             print(count+1,char)
                count = count + 1
                t.append(count)


            jilu.append(t)
    #     print('jilu:',jilu)

        for step3, item in enumerate(new_dep_outputs):
            if item[0] != 'ROOT':
                new_dep_outputs[step3] = (item[0],jilu[item[1]-1][1:],jilu[item[2]-1][1:])
            else:
                new_dep_outputs[step3] = (item[0],[item[1]],jilu[item[2]-1][1:])
    #             print(item[2]-1)
        # print('new_dep_outputs',new_dep_outputs)

        final_dep_outputs = []
        for item in new_dep_outputs:
            if item[0] != 'ROOT':
                for j in item[1]:
                    for k in item[2]:
                        final_dep_outputs.append((item[0],j,k))

                for j in item[2]:
                    for k in item[1]:
                        final_dep_outputs.append((item[0],k,j)) 
            else:
                for j in item[1]:
                    for k in item[2]:
                        final_dep_outputs.append((item[0],j,k))
    #     print(final_dep_outputs)

    #     length = len(''.join(tokens))
        # transfomer的最后面需要补个endMAX_LENGTH+1
        dep = np.zeros((MAX_LENGTH+1, MAX_LENGTH+1), dtype=np.float32)
        for item in final_dep_outputs:
            if item[0] != 'ROOT':
                dep[item[1],item[2]]=1
                dep[item[2],item[1]]=1
    #     print(dep.shape)
    #     print(dep)
        if (dep.transpose() == dep).all() and length_item == length_tokens:
            pass
        else:
            print('处理错误：', step, ''.join(tokens))
            dep = np.zeros((MAX_LENGTH+1, MAX_LENGTH+1), dtype=np.float32)
    #     print(dep.shape)

        dep = torch.from_numpy(dep).unsqueeze(0)
        pickle.dump(dep,save_file)

    save_file.close()
    nlp.close()
    return True

# os.system('ps -ef | grep java | grep -v grep | cut -c 9-15 | xargs kill -9')
# path = r"/home/chengkun/jupyter_projects/Magic-NLPer-main/data/data_sets/dep_val_pairs"  # 文件路径
# if os.path.exists(path):  # 如果文件存在
#     # 删除文件，可使用以下两种方法。
#     os.remove(path)  
#     #os.unlink(path)
# else:
#     print('未找到文件')  # 则返回文件不存在
# save_file = open(path, "wb")
# f1(val_pairs,save_file)

# load_file.close()
#（‘ROOT’,0,2）表示第二个词（like）是“I like swimming .”这句话的根节点；（‘nsubj’，2，1）表示第1个词（I）的父节点（也就是它的head）是第2个词like


# In[ ]:


os.system('ps -ef | grep java | grep -v grep | cut -c 9-15 | xargs kill -9')
path = current_datasets_path + "dep_train_pairs"  # 文件路径
if os.path.exists(path):  # 如果文件存在
    # 删除文件，可使用以下两种方法。
    os.remove(path)  
    #os.unlink(path)
else:
    print('未找到文件')  # 则返回文件不存在  

save_file = open(path, "wb")
f1(train_pairs,save_file)


# In[ ]:


os.system('ps -ef | grep java | grep -v grep | cut -c 9-15 | xargs kill -9')
print('运行结束')


# In[ ]:


# load_file = open("/home/chengkun/jupyter_projects/Magic-NLPer-main/data/dep.bin", "rb")
# data = pickle.load(load_file)
# load_file.close()


# In[2]:


# import torch
# import datetime
# final_dep = torch.zeros(1,102,102)
# for i in range(500000):
#     if i%1000==0:
#         print(final_dep.shape)
#         time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         print(time2)
#     a = torch.zeros(1,102,102)
#     final_dep = torch.cat((final_dep,a),0)
# print(final_dep.shape)


# In[ ]:


# import torch
# import pickle
# import time
# A = np.zeros((1,2,2), dtype=np.float32)
# B = np.ones((1,2,2), dtype=np.float32)
# A=torch.from_numpy(A)    #2x3的张量（矩阵）
# print(A)
# B=torch.from_numpy(B)  #4x3的张量（矩阵）
# print(B)
# C=torch.cat((A,B),0)  #按维数0（行）拼接
# # print(C.shape)

# save_file = open("/home/chengkun/jupyter_projects/Magic-NLPer-main/data/dep.bin", "wb")
# load_file = open("/home/chengkun/jupyter_projects/Magic-NLPer-main/data/dep.bin", "rb+")
# pickle.dump(C, save_file)

# data = pickle.load(load_file)
# print(data)
# save_file.close()
# load_file.close()

