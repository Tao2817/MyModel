import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, SAGEConv, GATConv, GAT
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time, random, datetime, os
from utils.train import train, draw_pics


######
## attention前qkv才relu         
## embedding cat的时候不要relu   DONE



global se, sw, epoch, args
# class GIN(nn.Module):
#     def __init__(self, nfeat, n_se, nhid, nclass, nlayer, dropout):
#         super().__init__()
#         self.num_layers = nlayer
#         self.dropout = dropout

#         self.pre = nn.Sequential(nn.Linear(nfeat, nhid))

#         self.embedding_s = nn.Linear(n_se, nhid)

#         self.graph_convs = nn.ModuleList()
#         self.nn1 = nn.Sequential(nn.Linear(nhid + nhid, nhid), nn.ReLU(), nn.Linear(nhid, nhid))
#         self.graph_convs.append(GINConv(self.nn1))
#         self.graph_convs_s_gcn = nn.ModuleList()
#         self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))

#         for l in range(nlayer - 1):
#             self.nnk = nn.Sequential(nn.Linear(nhid + nhid, nhid), nn.ReLU(), nn.Linear(nhid, nhid))
#             self.graph_convs.append(GINConv(self.nnk))
#             self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))

#         self.Whp = nn.Linear(nhid + nhid, nhid)
#         self.post = nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU())
#         self.readout = nn.Sequential(nn.Linear(nhid, nclass))
#     #     self.init()
        
#     # def init(self,):
#     #     for m in self.children():
#     #         if isinstance(m,nn.Sequential)
#     #             pass
#     #         nn.init.kaiming_normal(m.weight.detach())
#     #         if m.bias != None:
#     #             m.bias.detach().zero_()

#     def forward(self, data):
#         x, edge_index, batch, s = data.x, data.edge_index, data.batch, data.stc_enc
#         x = self.pre(x)
#         s = self.embedding_s(s)
#         for i in range(len(self.graph_convs)):
#             x = torch.cat((x, s), -1)
#             x = self.graph_convs[i](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, self.dropout, training=self.training)
#             s = self.graph_convs_s_gcn[i](s, edge_index)
#             s = torch.tanh(s)

#         x = self.Whp(torch.cat((x, s), -1))
#         x = global_add_pool(x, batch)
#         x = self.post(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.readout(x)
#         x = F.log_softmax(x, dim=1)
#         return x

class AddLinear(nn.Module):
    def __init__(self, in_dim) -> None:
        super(AddLinear, self).__init__()
        self.linear = nn.Linear(in_dim * 2, in_dim)
        self.linear.weight = nn.Parameter(torch.cat([torch.eye(in_dim), torch.eye(in_dim)], dim=1), requires_grad=True)
        self.linear.bias = nn.Parameter(torch.zeros(in_dim), requires_grad=True)

    def forward(self, inputs):
        input_1, input_2 =inputs
        return self.linear(torch.cat([input_1, input_2], -1))

class SpacialEmbeddingFuse(nn.Module):
    def __init__(self, embed_num ,embed_dim) -> None:
        super(SpacialEmbeddingFuse, self).__init__()
        self.embed_num, self.embed_dim = embed_num, embed_dim
        self.fuse_layer = nn.Sequential(nn.Linear(embed_num,32),
                                        nn.ReLU(),
                                        nn.Linear(32,1))
        self.score_layer = nn.Sequential(nn.Linear(embed_dim, embed_dim*2),
                                         nn.ReLU(),
                                         nn.Linear(embed_dim*2, 1))

    def forward(self, spacial_embeddings:torch.Tensor)->torch.Tensor:
        # # (embed_num,node_num,embed_dim) -> (node_num,embed_num,embed_dim) 
        # spacial_embeddings = spacial_embeddings.permute(1,0,2)
        # # (node_num,embed_num,embed_dim) -> (node_num,embed_num,1)
        # scores = self.score_layer(spacial_embeddings)
        # # (node_num,embed_dim,embed_num) @ (node_num,embed_num,1) = (node_num,embed_dim,1)
        # fused_se = spacial_embeddings.permute(0,2,1) @ F.softmax(scores,dim=1)  
        #region old fuse
        # (embed_num,node_num,embed_dim) -> (node_num,embed_dim,embed_num) 
        spacial_embeddings = spacial_embeddings.permute(1,2,0)
        #(node_num,embed_dim,embed_num)  -> (node_num,embed_dim,1)
        fused_se = (self.fuse_layer(spacial_embeddings)) + spacial_embeddings[:,:,18].unsqueeze(-1)
        # fused_se = self.fuse_layer(spacial_embeddings)
        #endregion
        #region no fuse
        # fused_se = spacial_embeddings[:,:,18].unsqueeze(-1)
        #endregion
        # (node_num,embed_dim,1)
        return fused_se.permute(2,0,1)
    
class STEmb(nn.Module):
    def __init__(self, s_dim, t_dim, ST_hid_dim) -> None:
        super().__init__()
        self.s_dim = s_dim
        self.t_dim = t_dim
        self.fc_s = nn.Linear(s_dim, ST_hid_dim)
        self.fc_t = nn.Linear(t_dim, ST_hid_dim)
        self.fuse_st = nn.Linear(ST_hid_dim*2,ST_hid_dim)
    def forward(self, se, te) -> torch.Tensor:
        # s: 64->64, t:307->64
        # se: (batch_size, node_num, 64) te:(batch_size, his_len+fut_len, 64)
        # ste: (batch_size, his_len+fut_len, node_num, 64)
        # se, te = F.relu(self.fc_s(se)), F.relu(self.fc_t(te))
        se, te = self.fc_s(se), self.fc_t(te)

        mean = se.mean(dim=-1,keepdim=True)
        std = se.std(dim=-1,keepdim=True)
        se = se - mean
        se = se / std
        mean = te.mean(dim=-1,keepdim=True)
        std = te.std(dim=-1,keepdim=True)
        te = te - mean
        te = te / std

        _, node_num, hid_dim = se.shape
        batch_size, his_fut_len, _, hid_dim = te.shape

        se = se.reshape(1,1,node_num,hid_dim).repeat(batch_size,his_fut_len,1,1)
        te = te.repeat(1,1,node_num,1)
        #
        st_emb = self.fuse_st(F.relu(torch.cat([se, te], dim=-1)))
        # st_emb = F.relu(self.fuse_st(torch.cat([se, te], dim=-1)))
        return st_emb
    
class BaseGen(nn.Module):
    def __init__(self,args) -> None:
        super(BaseGen, self).__init__()
        self.args = args
        self.base_his_proj = nn.Sequential(nn.Linear(args.his_len, args.hid_dim),
                                           nn.ReLU(),
                                           nn.LayerNorm(args.hid_dim),
                                           nn.Linear(args.hid_dim,args.his_len))
        #
        self.x_proj = nn.Sequential(nn.Linear(args.hid_dim + args.hid_dim//2, args.hid_dim),
                                    nn.ReLU())
        self.base_fut_proj = nn.Sequential(nn.Linear(args.his_len, args.hid_dim),
                                           nn.ReLU(),
                                           nn.LayerNorm(args.hid_dim),
                                           nn.Linear(args.hid_dim,args.fut_len))
        #
        self.y_proj = nn.Sequential(nn.Linear(args.hid_dim + args.hid_dim//2, args.hid_dim),
                                    nn.ReLU())
        # self.base_proj = nn.Sequential(nn.Linear(args.his_len, args.base_num),
        #                               nn.ReLU(),
        #                               nn.Linear(args.base_num,args.his_len+args.fut_len))
        # self.x_proj = nn.Linear(args.hid_dim * 2, args.hid_dim)
    def forward(self,x,ste):
        # x:(his, node_num, hid_dim)
        # ste:(batch_size, his+fut, node_num, hid_dim)
        ste_his, ste_fut = torch.split(ste,[12,12],dim=1)
        # (batch_size, his,node_num, 2*hid_dim)
        x_ste = torch.cat([x,ste_his],dim=-1)
        y_ste = torch.cat([x,ste_fut],dim=-1)
        # (batch_size, his, node_num, hid_dim),64 or base_num?
        x_ste = self.x_proj(x_ste)
        # (batch_size, node_num, hid_dim, his)
        x_ste = x_ste.permute(0,2,3,1)
        # (node_num,hid_dim, his),(207, 128, 12)
        base_his = self.base_his_proj(x_ste)

        y_ste = self.y_proj(y_ste)
        y_ste = y_ste.permute(0,2,3,1)
        base_fut = self.base_fut_proj(y_ste)
        # (node_num,hid_dim,his)
        return base_his, base_fut
    

        # # x:(his, node_num, hid_dim)
        # # ste:(batch_size, his+fut, node_num, hid_dim)
        # ste_his, ste_fut = torch.split(ste,[12,12],dim=1)
        # # (batch_size, his,node_num, 2*hid_dim)
        # x_ste = torch.cat([x,ste_his+ste_fut],dim=-1)
        # # (batch_size, his, node_num, hid_dim),64 or base_num?
        # x_ste = self.x_proj(x_ste)
        # # (batch_size, node_num, hid_dim, his)
        # x_ste = x_ste.permute(0,2,3,1)
        # # (node_num,hid_dim, his+fut),(207, 128, 24)
        # base = self.base_proj(x_ste)
        # base_his, base_fut = torch.split(base,[12,12],dim=-1)
        # # (node_num,hid_dim,his)
        # return base_his, base_fut

import math
from torch import nn
import torch
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型的维度
        self.d_k = d_model // heads  # 每个头的维度
        self.h = heads  # 头的数量

        # 以下三个是线性层，用于处理Q（Query），K（Key），V（Value）
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.out = nn.Linear(d_model, d_model)  # 输出层

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        # torch.matmul是矩阵乘法，用于计算query和key的相似度
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)  # 在第一个维度增加维度
            scores = scores.masked_fill(mask == 0, -1e9)  # 使用mask将不需要关注的位置设置为一个非常小的数

        # 对最后一个维度进行softmax运算，得到权重
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)  # 应用dropout

        output = torch.matmul(scores, v)  # 将权重应用到value上
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)  # 获取batch_size

        # 将Q, K, V通过线性层处理，然后分割成多个头
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 转置来获取维度为bs * h * sl * d_model的张量
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 调用attention函数计算输出
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # 重新调整张量的形状，并通过最后一个线性层
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)  # 最终输出
        return output


class Atten(nn.Module):
    def __init__(self,args) -> None:
        super(Atten, self).__init__()
        self.args = args
        self.wQ = nn.Linear(args.hid_dim,args.hid_dim)
        self.wK = nn.Linear(args.hid_dim,args.hid_dim)
        self.wV = nn.Linear(args.hid_dim,args.hid_dim)
        # self.atten = nn.MultiheadAttention(args.hid_dim, args.atten_num_heads,batch_first=True,dropout=0.1)
        self.output = nn.Linear(args.hid_dim,args.hid_dim)

    def forward(self,q,k,v):
        Q = F.relu(self.wQ(q))
        K = F.relu(self.wK(k))
        V = F.relu(self.wQ(v))

        # attention




        x_, attention_score= self.atten(Q,K,V)
        return self.output(x_), attention_score
    
# region baseattn
# class BaseAtten(nn.Module):
#     def __init__(self, args) -> None:
#         super(BaseAtten, self).__init__()
#         self.layers = nn.ModuleList()
#         self.base_gen = BaseGen(args)
#         self.atten = Atten(args)
#         # self.base_gen = BaseGen(args)
#         # self.base_atten = Atten(args)
#         # self.gcns = GCNConv()
    
#     def forward(self, x, ste):
#         # (12,207,64) 
#         # x:(batch_size, node_num, hid_dim, his_len)
#         # base_his:(batch_size, node_num, hid_dim, his_len)
#         base_his, base_fut = self.base_gen(x,ste)
#         batch_size, node_num, hid_dim, his_len = base_his.shape
#         batch_size, node_num, hid_dim, fut_len = base_fut.shape
        
#         # attention on last 2 dim, e.q. x(hid_dim, his_len) <-> base(hid_dim,his_len)
#         x = x.reshape(batch_size * node_num, hid_dim, his_len)
#         base_his = base_his.reshape(batch_size * node_num, hid_dim, his_len)
#         base_fut = base_fut.reshape(batch_size * node_num,hid_dim, fut_len)

#         # (207,64,12)
#         x_, atten_score = self.atten(x, base_his, base_fut)

#         # turn back to original dim
#         x = x.reshape(batch_size, node_num, hid_dim, his_len)
#         x_ = x_.reshape(batch_size, node_num, hid_dim, his_len)
#         atten_score = atten_score.reshape(batch_size, node_num, hid_dim, hid_dim)
#         base_his = base_his.reshape(batch_size, node_num, hid_dim, his_len)
#         base_fut = base_fut.reshape(batch_size, node_num,hid_dim, fut_len)

#         return x_, atten_score, base_his, base_fut
# endregion


class TimeAttn(nn.Module):
    def __init__(self, args) -> None:
        super(TimeAttn, self).__init__()
        self.hid_dim = args.hid_dim
        self.time_dim = 288
        self.time_memory = self.init_memory()

        self.dropout = nn.Dropout(args.drop_out)
        self.lin_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.lin_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.lin_v = nn.Linear(self.hid_dim, self.hid_dim)

    def init_memory(self):
        memory = nn.Parameter(torch.randn(self.time_dim, self.hid_dim), requires_grad=True)     # (M, d)
        nn.init.xavier_normal_(memory)
        return memory

    def attention_forward(self, q, k, v):
        
        # torch.matmul是矩阵乘法，用于计算query和key的相似度
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hid_dim ** 0.5)

        # if mask is not None:
        #     mask = mask.unsqueeze(1)  # 在第一个维度增加维度
        #     scores = scores.masked_fill(mask == 0, -1e9)  # 使用mask将不需要关注的位置设置为一个非常小的数

        # 对最后一个维度进行softmax运算，得到权重
        scores = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            scores = self.dropout(scores)  # 应用dropout

        output = torch.matmul(scores, v)  # 将权重应用到value上
        return output, scores

    def forward(self, x:torch.Tensor):
        x_org = x
        x = x.permute(0,2,1,3)

        # batch_size, node_num, time_len, hid_dim = x.shape

        q = self.lin_q(x)
        k = self.lin_k(self.time_memory)
        v = self.lin_v(self.time_memory)

        # q = torch.cat(torch.split(q, self.d, dim=-1), dim=0)
        # k = torch.cat(torch.split(k, self.K, dim=-1), dim=0)
        # v = torch.cat(torch.split(v, self.K, dim=-1), dim=0)

        # torch.matmul是矩阵乘法，用于计算query和key的相似度
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / (self.hid_dim ** 0.5)

        # score_his, score_fut = torch.split(attn_score, [12,12], dim=2)

        # 对最后一个维度进行softmax运算，得到权重, 历史与未来分别计算
        # score_his = F.softmax(score_his, dim=-1)
        # score_fut = F.softmax(score_fut, dim=-1)

        attn_score = F.softmax(attn_score, dim=-1)

        if self.dropout is not None:
            attn_score = self.dropout(attn_score)  # 应用dropout

        x = torch.matmul(attn_score, v).permute(0,2,1,3)

        # x, attn_score = self.attention_forward(q, k, v)

        _, ind = torch.topk(attn_score, k=2, dim=-1)
        # 只对真实的x的注意力分数取样，y是预测值不使用
        # 正样本，memory中与x最相似的 
        # [batch_size, node_num, his_len, 64]
        pos = self.time_memory[ind[:, :, :, 0]].permute(0,2,1,3)
        # 负样本，第二相似 
        neg = self.time_memory[ind[:, :, :, 1]].permute(0,2,1,3)

        return x, x_org, pos, neg
    

class STLayer(nn.Module):
    def __init__(self, args) -> None:
        super(STLayer, self).__init__()
        self.s_layer = GCNConv(args.hid_dim,args.hid_dim) 
        self.t_layer = nn.MultiheadAttention(args.hid_dim, args.atten_num_heads, args.drop_out, batch_first=True)
        self.fuse_layer = nn.Sequential(AddLinear(args.hid_dim),
                                        nn.ReLU(),
                                        nn.Linear(args.hid_dim, args.hid_dim))
        
    def forward(self, x, edge_list):
        # spacial
        x_s = self.s_layer(x, edge_list)

        # temporal
        # attention on (time, hid_dim)
        batch_size, his_len, node_num, hid_dim = x.shape
        
        x_t = x.permute(0,2,1,3).reshape(batch_size * node_num, his_len, hid_dim)
        x_t, attention_score = self.t_layer(x_t, x_t, x_t)
        x_t = x_t.reshape(batch_size, node_num, his_len, hid_dim).permute(0,2,1,3)

        # fuse
        x_st = self.fuse_layer((x_s, x_t))
        # x_st = x_s
        # return x_st + x
        return x_st

class MyModel(nn.Module):
    def __init__(self, args) -> None:
        super(MyModel, self).__init__()
        self.args = args
        self.feature_dim, self.hid_dim = args.feature_dim, args.hid_dim
        self.embed_dim, self.embed_num = args.S_embed_dim, args.S_embed_num
        self.x_pre = nn.Sequential(nn.Linear(self.feature_dim, self.hid_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hid_dim, self.hid_dim))
        
        self.te_memory_fuse = nn.Linear(288, args.his_len, bias=False)

        self.se_fuse_layer = SpacialEmbeddingFuse(self.embed_num, self.embed_dim)

        self.st_embed_layer = STEmb(args.S_dim, args.T_dim, args.ST_hid_dim)

        self.ste_fuse_layer_his = nn.Sequential(AddLinear(args.hid_dim),
                                                nn.ReLU(),
                                                nn.Linear(args.hid_dim, args.hid_dim))
                                        # _BatchNorm(args.hid_dim),
        # # initiate the linear(x, y) to x + y
        # self.ste_fuse_layer_his.weight = nn.Parameter(torch.cat([torch.eye(args.hid_dim), torch.eye(args.hid_dim)], dim=1), requires_grad=True)
        # self.ste_fuse_layer_his.bias = nn.Parameter(torch.zeros(args.hid_dim), requires_grad=True)

        self.ste_fuse_layer_fut = AddLinear(args.hid_dim)
                                        # _BatchNorm(args.hid_dim),
        
        self.memory_fuse_layer_his = AddLinear(args.hid_dim)

        # # initiate the linear(x, y) to x + y
        # self.ste_fuse_layer_fut.weight = nn.Parameter(torch.cat([torch.eye(args.hid_dim), torch.eye(args.hid_dim)], dim=1), requires_grad=True)
        # self.ste_fuse_layer_fut.bias = nn.Parameter(torch.zeros(args.hid_dim), requires_grad=True)

        #gcn->gin
        # self.nn1 = torch.nn.Sequential(torch.nn.Linear(args.hid_dim, args.hid_dim), torch.nn.ReLU(), torch.nn.Linear(args.hid_dim, args.hid_dim))
        # self.gcns = nn.ModuleList([GINConv(self.nn1) for _ in range(args.layer_num_g)])

        self.encoder = nn.ModuleList([STLayer(args) for _ in range(args.layer_num)])

        self.memory_fuse_layer_fut = AddLinear(args.hid_dim)

        self.time_memory_attn = TimeAttn(args)

        self.decoder = nn.ModuleList([STLayer(args) for _ in range(args.layer_num)])

        self.output = nn.Linear(args.hid_dim,args.feature_dim)

    
    def forward(self, x:torch.Tensor, se:torch.Tensor, te:torch.Tensor, edge_list) -> torch.Tensor:
        se = self.se_fuse_layer(se)#(embed_num,embed_dim)->(1,embed_dim)

        # (24,64)
        te_memory = self.te_memory_fuse(self.time_memory_attn.time_memory.transpose(1,0)).transpose(1,0).unsqueeze(1)
        # ste:(batch_size, his+fut, node_num, hid_dim)
        ste = self.st_embed_layer(se,te)
        # (batch_size, his or fut, node_num, hid_dim)
        ste_his, ste_fut = torch.split(ste,[self.args.his_len,self.args.his_len],dim=1)

        # (batch_size, his, node_num, feature_dim)->(batch_size, his, node_num, hid_dim)
        x = self.x_pre(x)#(12,207,1)->(12,207,64)


        x = self.ste_fuse_layer_his((x, ste_his))
        # x_memory_his, x_org, pos, neg = self.time_memory_attn(x)
        # x = self.memory_fuse_layer_his((x, x_memory_his))

        for layer in self.encoder:
            x = layer(x, edge_list)

        x = self.ste_fuse_layer_fut((x, ste_fut))
        x_memory_fut, x_org, pos, neg = self.time_memory_attn(x)
        x = self.memory_fuse_layer_fut((x, x_memory_fut))
        
        for layer in self.decoder:
            x = layer(x, edge_list)
        # (12,207,64)
        y_ = self.output(x)
        return y_, x_org, pos, neg
    
    def mae(self, x, x_):
        mae_loss = torch.abs(x-x_).mean(dim=1).mean(dim=1).sum()
        return mae_loss
    
    def mse_loss(self, x, x_):
        return F.mse_loss(x, x_)
        
    def masked_mae_loss(self, preds, labels, null_val=np.nan):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels != null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds - labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)
    
    def loss(self, x, x_):
        return self.mse_loss(x, x_)


if __name__ == "__main__":
    from tensorboardX import SummaryWriter
    # torch.set_default_device("cuda")
    import torch.optim as optim

    from utils.data2 import load_multi_domain
    torch.manual_seed(1840130100400)
    
    class dummyArgs():
        def __init__(self) -> None:
            self.data_path = "./traffic_dataset"
            self.dataset_names = ["METR_LA"]
            self.current_dataset = "METR_LA"
            self.SE_nums = 25
            self.aggregation_rate = 0.1
            self.aggregation_size = 3
            self.shrink_rate = 1
            self.batch_size = 64
            self.hid_dim = 64
            self.feature_dim = 1
            self.S_embed_dim = 64
            self.S_embed_num = 25
            self.S_dim = 64
            self.base_num = 16
            # 288 + 12 + 7 = 307
            self.T_dim = 307
            self.ST_hid_dim = 64
            self.node_num = 207
            self.atten_num_heads = 4
            self.drop_out = 0.1
            self.his_len = 12
            self.fut_len = 12
            self.layer_num = 3
            self.layer_num_t = 3
            self.layer_num_g = 3
            self.epoch_num = 300
            self.save_path = f"./runs/model_test/{self.current_dataset}/test_{datetime.datetime.today().strftime(r'%Y-%m-%d-%H-%M')}"
            self.resume = False
            self.resume_epoch = 35
            self.checkpoint_path = f"./runs/model_test/{self.current_dataset}/check_point/model layernorm"
            self.lr = 2e-3

    seed = torch.initial_seed()
    
    matplotlib.use("agg")
    args = dummyArgs()
    args.save_path = f"{args.save_path}_seed-{seed}"
    my_model = MyModel(args).cuda()
    sw = SummaryWriter(args.save_path)
    print(f"total params: {sum(param.nelement() for param in my_model.parameters())}")
    #region dummy input
    # x = torch.randn((12,args.node_num,1)) #(his+fut,node_num,feature_dim)
    # te = torch.randn((24,args.T_dim)) #(his+fut,hid_dim)
    # se = torch.tensor([list(range(args.S_embed_dim))for _ in range(args.S_embed_num)]).to(torch.float32)
    # edge_list = torch.randint(207,(2,1000))
    #endregion

    lossf = nn.MSELoss().cuda()
    # scheduler = optim.lr_scheduler.StepLR(optimizer,
    #                                     step_size=10,
    #                                     gamma=0.9)

    multi_datasets = load_multi_domain(args)
    dataset = multi_datasets[args.current_dataset]

    (train_loader, vali_loader, test_loader),se = dataset
    se = se.cuda()
    if args.resume:
        #start from this epoch
        try:
            start_epoch = args.resume_epoch
            print(f"resume from epoch {start_epoch}")
            checipoint = torch.load(f"{args.checkpoint_path}/test-epoch{start_epoch}.pkl")
            start_epoch = checipoint["epoch"] + 1
            my_model.load_state_dict(checipoint["model"])
            optimizer = optim.AdamW(my_model.parameters(), args.lr)
            optimizer.load_state_dict(checipoint["optimizer"])
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=1,
                                                  gamma=0.9)
            # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=args.resume_epoch if args.resume else -1)
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-4, last_epoch=args.resume_epoch if args.resume else -1)
            scheduler.load_state_dict(checipoint["scheduler"])
        except:
            print("model loading failed, start from epoch 0")
            start_epoch = 0
            optimizer = optim.AdamW(my_model.parameters(),args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=1,
                                                  gamma=0.9)
            # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-4, last_epoch=args.resume_epoch if args.resume else -1)
            pass
    else:
        #train from epoch 0
        optimizer = optim.AdamW(my_model.parameters(), args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=1,
                                                  gamma=0.9)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=args.resume_epoch if args.resume else -1)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-4, last_epoch=args.resume_epoch if args.resume else -1)

        start_epoch = 0

    for epoch in range(start_epoch, args.epoch_num):

        print(f"epoch: {epoch} start")
        time_1 = time.time()
        my_model.train()
        print("\ttrain start")
        all_batch_y_pred_train, loss_train, (mae_y, rmse_y, mape_y) = train("train", train_loader, my_model, se, optimizer, scheduler)
        sw.add_scalar(f"train-loss",loss_train,epoch)
        sw.add_scalar(f"train-lr",scheduler.get_last_lr(),epoch)
        sw.add_scalars(f"train-metrics",{"mae_y":mae_y,"rmse_y":rmse_y,"mape_y":mape_y},epoch)
        print(f'\ttrain finished in {time.time()-time_1} s')
        
        
        
        
        # save model weights
        if not  os.path.exists(f"{args.checkpoint_path}"):
            os.mkdir(f"{args.checkpoint_path}")
        torch.save({
            "epoch":epoch,
            "model":my_model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict(),
            }, f"{args.checkpoint_path}/test-epoch{epoch}.pkl")
        print(f"\tepoch: {epoch} model saved")
        
        
        
        
        time_1 = time.time()
        with torch.no_grad():
            my_model.eval()
            print("\tvali start")
            all_batch_y_pred_vali, loss_vali, (mae_y, rmse_y, mape_y) = train("vali", vali_loader, my_model, se)
            sw.add_scalar(f"vali-loss",loss_vali,epoch)
            sw.add_scalars(f"vali-metrics",{"mae_y":mae_y,"rmse_y":rmse_y,"mape_y":mape_y},epoch)
            print(f'\tvali finished in {time.time()-time_1} s')
            time_1 = time.time()
            print("\ttest start")
            all_batch_y_pred_test, loss_test, (mae_y, rmse_y, mape_y) = train("test", test_loader, my_model, se)
            sw.add_scalar(f"test-loss",loss_test,epoch)
            sw.add_scalars(f"test-metrics",{"mae_y":mae_y,"rmse_y":rmse_y,"mape_y":mape_y},epoch)
            print(f'\ttest finished in {time.time()-time_1} s')
            time_1 = time.time()

        # drawing pictures
        # torch.save((rand_batch_train, all_batch_y_pred_train,rand_batch_vali, all_batch_y_pred_vali,rand_batch_test, all_batch_y_pred_test),
        #            f"./testdata.pkl")
        draw_pics(args, all_batch_y_pred_train, "train", epoch)
        draw_pics(args, all_batch_y_pred_vali, "vali", epoch)
        draw_pics(args, all_batch_y_pred_test, "test", epoch)

        pass
    sw.close()

        