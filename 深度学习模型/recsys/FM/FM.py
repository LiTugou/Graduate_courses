import torch
import torch.nn as nn

class FM(nn.Module):
    def __init__(self,features_num,k):
        super().__init__()
        ## 一阶系数
        self.features_num=features_num
        self.k=k
        self.w0=torch.empty((1),requires_grad=True)
        self.w=torch.empty((features_num,1),requires_grad=True)
        self.v=torch.empty((features_num,k),requires_grad=True)
        nn.init.normal_(self.w0,mean=0,std=0.1)
        nn.init.normal_(self.w,mean=0,std=0.1)
        nn.init.normal_(self.v,mean=0,std=0.1)


    def forward(self,X):
        # X (Batch,features)
        # X_view=X.expand((k,*X.shape))
        # tmp=self.vfunc(self.v,X.expand())
        reg=self.w0+X@self.w
        for f in range(self.k):
            reg+=FM.cross_vx(self.v[:,f],X)
        return reg/2

    def cross_vx(v,X):
        """
        X (Batch,features)
        v (features,1)
        """
        tmp=v.view((1,-1))*X
        return torch.sum((tmp.sum(axis=1))**2)-torch.sum(tmp**2)
    
    
class DenseEmbedding(nn.Module):
    def __init__(self,field_features,factory_num,sparse_start):
        super().__init__()
        self.sparse_start=sparse_start
        self.field_num=len(field_features)
        self.embedding_layer=nn.ModuleList([
            nn.Embedding(field_features[i],factory_num,sparse=True)
                for i in range(self.field_num)
        ])
        self.features_range=[]
        start=sparse_start
        for i in range(self.field_num):
            end=start+field_features[k]
            rng=(start,end)
            self.features_range.append(rng)
            start=end
            
    def forward(self,X):
        res=[X[:,:sparse_start]]
        for i in range(self.field_num):
            start,end=self.features_range[i]
            embedding=self.embedding_layer[i]
            tmp=embedding(X[:,start:end])
            res.append(tmp)
        return torch.cat(res,axis=1)

class FNN(nn.Module):
    def __init__(self,field_features,features_num,sparse_start,factory_num=100):
        # sparse_start 稀疏特征开始的索引
        # field_feature 每个field有多少个feature
        # features_num 总共有多少feature
        # fac_num 因子的维度

        super().__init__()
        self.field_num=len(field_feature)
        self.features_num=features_num
        self.factory_num=factory_num
        self.fm_model=FM(features_num,fac_num)
        
        # Dense Embedding
        self.dense_embedding=DenseEmbedding(field_features,factory_num,sparse_start)
        # DNN
        self.DNN=nn.Sequential(
            ["fc1",nn.Linear(factory_num*self.field_num,1024)],
            ["ac1",nn.Tanh()],
            ["fc2",nn.Linear(1024,1024)],
            ["ac2",nn.Tanh()],
        )
        self.output=nn.Sequential(
            ["fc_out",nn.Linear(1024,1)],
            ["ac_out",nn.Sigmoid()]
        )
        
    def pretrain(self,X):
        pass
        # self.w0=self.fm_model.w0
        # self.w=self.fm_model.w
        # self.v=self.fm_model.v
        # use pretrain v to init embedings weights
    
    # 没有标记连续型变量
    def forward(self,X):
        # X (Batch,features)
        # feature_range list of tuple ,(start,end)
        X=self.dense_embeding(X)
        X=self.DNN(X)
        X=self.output(X)
        return X

class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        ## 一阶系数
        self.field_num=len(field_feature)
        self.features_num=features_num
        self.factory_num=factory_num
        self.fm_model=FM(features_num,fac_num)
        
        # Dense Embedding
        self.dense_embedding=DenseEmbedding(field_features,factory_num,sparse_start)
        # DNN
        self.DNN=nn.Sequential(
            ["fc1",nn.Linear(factory_num*self.field_num,1024)],
            ["ac1",nn.Tanh()],
            ["fc2",nn.Linear(1024,1024)],
            ["ac2",nn.Tanh()],
        )
        self.output=nn.Sequential(
            ["fc_out",nn.Linear(1024,1)],
            ["ac_out",nn.Sigmoid()]
        )
    
# FM Recsys
# FFM
# 每个特征i 对不同field的特征j;fi表示特征i属于的field 有不同的v(v_ifj) ,所以二阶交互是 <v_ifj,v_jfi>
# https://www.jianshu.com/p/43d1976dfe44
# FNN
# https://blog.csdn.net/qq_38375203/article/details/125266753
# https://www.jianshu.com/p/c639d52c124b
# https://jesse-csj.github.io/2019/07/21/FNN/