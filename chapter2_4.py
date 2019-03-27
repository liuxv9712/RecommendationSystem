import math
from operator import itemgetter


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
#但是上面那种方法的复杂度为O（|U|*|U|）,这在用户数很大时非常耗时。很多时候|N(u) & N(v)| = 0。因此改进算法：
def UserSimilarityUpdate(train):
    #build inverse table for item_users建立物品到用户的倒排表
    item_users = dict()
    for u,items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    #calculate co-rated items between users
    C = dict()
    N = dict()
    for i,users in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                C[u][v] += 1

    #calculate finial similarity matrix W
    W = dict()
    for u,related_users in C.items():
        for v,cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

# 实现UserCF推荐算法
def RecommendUserCF(user, train, W, K, u):
    rank = dict()
    interacted_items = train[user]
    for v,wuv in sorted(W[u].items,key=itemgetter(1),reverse=True)[0:K]:
        for i,rvi in train[v].items():
            if i in interacted_items:
                #we should filter items user interacted before continue
                rank[i]+=wuv*rvi
    return rank

# 实现User-IIF算法，本算法在对用户兴趣相似度的计算上采用了不同的公式
def RecommendUserIIF(train):
    #build inverse table for item_users
    item_users = dict()
    for u,items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    #calculate co-rated items between users
    C = dict()
    N = dict()
    for i,users in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u==v:
                    continue
                C[u][v] += 1 / math.log(1 + len(users))

    #calculate finial similarity matrix W
    W = dict()
    for u,related_users in C.items():
        for v,cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

#实现物品相似度计算
def ItemSimilarity(train):
    #calculate co-rated users between items
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1
    #calculate finial similarity matrix W
    W =dict()
    for i,related_items in C.items():
        for j,cij in related_items.items():
            W[i][j] = cij /math.sqrt(N[i] * N[j])
    return W