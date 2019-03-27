import math
import random
import pandas as pd


def GetRecommendation(user, N):
    ratings = pd.read_csv('ratings.csv', index_col=None)
    return ratings

#将数据集随机分成训练集和测试集。这里每次实验选取不同的k(0<=k<=M-1)和相同的随机数种子，进行M次实验就可以得到M个不同的训练集和测试集，然后分别进行实验，用M次实验的平均值作为最后的评测指标。这样子可以防止过拟合的结果。
def SplitData(data,M,k,seed):
    test = []
    train = []
    random.seed(seed)
    for user,item in data:
        if random.randint(0,M) == k:
            test.append([user,item])
        else:
            train.append([user,item])
    return train,test

#对两两用户利用余弦相似度计算相似度
def UserSimilarity(train):
    W = dict()
    for u in train.keys():
        for v in train.keys():
            if u == v :
                continue
            W[u][v] = len(train[u] & train[v])
            W[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)
    return W

#召回率描述有多少比例的用户-物品评分记录包含在最终的推荐列表中。
def Recall(train,test,N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user,N)
        for item,pui in rank:
            hit += 1
        all += len(tu)
    return hit / (all * 1.0)

#准确率描述最终的推荐列表中有多少比例是发生过的用户-物品评分记录。
def Precision(train,test,N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user,N)
        for item,pui in rank:
            if item in tu:
                hit += 1
        all += N
    return hit / (all * 1.0)

#覆盖率表示最终的推荐列表中包含多大比例的物品
def Coverage(train,test,N):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = GetRecommendation(user,N)
        for item,pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)

#新颖度。如果推荐出的物品都很热门，说明推荐的新颖度较低。
def Popularity(train,test,N):
    item_popularity = dict()
    for user,items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = GetRecommendation(user,N)
        for item ,pui in rank:
            ret += math.log(1+item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret

if __name__ == '__main__':
    user=[]
    N = 1
    ratings = GetRecommendation(user,N)
    SplitData(ratings, 8, 5, 10)
    similarity = UserSimilarity(train)
    recallRate = Recall()
    precisionRate = Precision()
    coverageRate = Coverage()
    popularityRate = Popularity()
    print(similarity,recallRate,precisionRate,coverageRate,popularityRate)