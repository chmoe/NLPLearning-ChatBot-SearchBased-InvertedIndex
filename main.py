# ------------------单词切分---------------------
# %% 
import jieba
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# %%
data_path = "./clean_chat_corpus/" # 数据集路径
train_data_name = "xiaohuangji.tsv"
# %%
train_data = pd.read_csv(data_path + train_data_name, sep = '\t', header = None)
# エラー出た
# 结巴分词出现AttributeError: 'float' object has no attribute 'decode'错误
# https://my.oschina.net/u/4336279/blog/3569965
train_data = train_data.astype(str) 
# %%

# 去除标点符号
# 来源：https://github.com/fxsjy/jieba/issues/169#issuecomment-49504512
punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
# 对str/unicode
filterpunt = lambda input_str: ''.join(filter(lambda x: x not in punct, input_str))
def sentence_process(input_str):
    '''
    将传入的中文句子调用jieba分词
    返回使用空格拼接的句子
    '''
    return " ".join(list(jieba.cut(filterpunt(input_str))))
# %%
# qaa_list = [] # 问题答案list
# for i in range(len(train_data[0])):
#     qaa_list.append([train_data[0][i], train_data[1][i]])
# %%
q_list = train_data[0].tolist() # 问题list
a_list = train_data[1].tolist() # 答案list
# %%
q_space_list = [sentence_process(i) for i in q_list]

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %% 训练
# 修改可以识别单个单词 https://blog.csdn.net/blmoistawinde/article/details/80816179
tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df = 0.6, stop_words=None)
# %%
train_v = tv.fit_transform(q_space_list) # 使用问题集训练

# %%
# dic_index = {}
# space_list = q_space_list[0:5]
# len_list = len(space_list)
# for index in range(len_list):
#     for word in space_list[index].split():
#         if word not in dic_index:
#             dic_index[word] = [index]
#         else:
#             dic_index[word].append(index)
# # %%
# dic_index["你好"]
# # %%
# q_space_list[355]
# %%
dic_index = {}
len_list = len(q_space_list)
str_len_list = str(len_list)
for index in range(len_list):
    print("创建倒排表： " + str(index) + "/" + str_len_list)
    for word in q_space_list[index].split():
        if word not in dic_index.keys(): # 单词不在倒排表中
            dic_index[word] = [index]
        else:
            dic_index[word].append(index)

# %%
def get_index(str_list, dic_index, threshold=1): # Threshold代表限制同时出现多少个单词的情况返回
    """
    根据输入的list
    和给定的字典倒排表
    输出所在的index
    """
    input_str = str_list[0]
    tmp_list = []
    for word in input_str.split():
        # print("正在查找单词："+word)
        if word in dic_index:
            for index in dic_index[word]:
                tmp_list.append(index)
    if(len(tmp_list)>10000):
        if(threshold > 1):
            len_word_list = len(input_str.split())
            threshold = threshold if threshold <= len_word_list else len_word_list
            tmp_list_choice = {}
            for item in tmp_list:
                if item in tmp_list_choice:
                    tmp_list_choice[item] += 1
                else:
                    tmp_list_choice[item] = 1
            tmp_list = []
            for k, v in tmp_list_choice.items():
                if v >= threshold:
                    tmp_list.append(k)
    return list(set(tmp_list))

# %%
def get_best_index_list(input_str):
    input_content = [sentence_process(input_str)]

    test_v = tv.transform(input_content)

    test_array = test_v.toarray()
    
    # tmp_cosine_sim = []
    # possible_index_list = get_index(input_content, dic_index, 3)
    # len_possible_index_list = len(possible_index_list)
    # str_len_possible_index_list = str(len_possible_index_list)
    # for index in range(len_possible_index_list):
    #     print("当前进度："+str(index)+" / "+str_len_possible_index_list)
    #     tmp_cosine_sim.append([float(cosine_similarity(train_v[possible_index_list[index]], test_array)), possible_index_list[index]])

    tmp_cosine_sim = []
    possible_index_list = get_index(input_content, dic_index, 1)
    len_possible_index_list = len(possible_index_list)
    for index in range(len_possible_index_list):
        tmp_cosine_sim.append([float(cosine_similarity(train_v[possible_index_list[index]], test_array)), possible_index_list[index]])

    # 排序筛选出来的内容
    tmp_return = sorted(tmp_cosine_sim, key=lambda x: x[0], reverse=True)

    return tmp_return
# %%
def get_qalist_with_list(input_str, range_int=1, output_q=False):
    tmp_return = get_best_index_list(input_str)
    result_list_len = len(tmp_return)
    if result_list_len != 0:
        range_int = result_list_len if result_list_len < range_int else range_int
        for i in range(range_int):
            if(output_q):print(str(i+1)+". 最匹配的问题："+q_list[tmp_return[i][1]])
            print(str(i+1)+". 其对应答案为："+a_list[tmp_return[i][1]])
    else:
        print("你在说些什么，本宝宝听不太懂呢")
# %%
get_qalist_with_list("我喜欢你",5,True)
# %%

# %%
# 效率优化问题
'''
scipy.sparse稀疏矩阵内积点乘--效率优化！：https://blog.csdn.net/mantoureganmian/article/details/80612137







'''
# %%
max_value_index = tmp_cosine_sim.index(max(tmp_cosine_sim))
# %%
print("最匹配的问题："+q_list[max_value_index])
print("其对应答案为："+a_list[max_value_index])

#%%
list(jieba.cut("我喜欢你"))
# %%
sentence_process("你好")
# %%
test_v_fnormal
# %%
possible_index_list
# %%
# tmp_cosine_sim = []
# possible_index_list = get_index(input_content, dic_index)
# len_possible_index_list = len(possible_index_list)
# str_len_possible_index_list = str(len_possible_index_list)
# 1
# import numpy as np

# test_v_fnormal = np.linalg.norm(test_array, 2) # 测试集的F范式


# for index in range(len_possible_index_list):
#     print("当前进度："+str(index)+" / "+str_len_possible_index_list)

#     cur_train_array = train_v[possible_index_list[index]].toarray()

#     cur_train_fnormal = np.linalg.norm(cur_train_array, 2) # 当前训练集的F范式

#     dot_result = cur_train_array * test_array

#     tmp_cosine_sim.append([(dot_result/(test_v_fnormal*cur_train_fnormal)), possible_index_list[index]])