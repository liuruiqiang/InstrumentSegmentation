# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
# from timm.models.layers import DropPath
from functools import partial
from timm.models.layers import DropPath, trunc_normal_


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim,inner_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(inner_dim,dim*2,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,z):
        B, N, C = x.shape
        B,NZ,CZ = z.shape
        q = self.q_proj(x)
        q = q.reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        kv = self.kv(z).reshape(B, NZ, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossTransformerBlock(nn.Module):

    def __init__(self, dim, inner_dim,num_heads,mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_z = norm_layer(inner_dim)
        self.attn = CrossAttention(
            dim,inner_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,ref_z):

        x = x + self.drop_path(self.attn(self.norm1(x),self.norm_z(ref_z)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# x = torch.rand(4,224*224,32).cuda()
# z = torch.rand(4,14*14,256).cuda()
# model = CrossTransformerBlock(dim=32,inner_dim=256,num_heads=8)
# model.cuda()
# out = model(x,z)
# print(out.shape)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class position_enc(nn.Module):
    def __init__(self,n_position, d_hid,drop_rate=0.2):
        super().__init__()
        self.n_position = n_position
        self.d_hid = d_hid
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=n_position, d_hid=d_hid), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = x+self.pos_embed
        x = self.pos_drop(x)
        return x

class learned_position_enc(nn.Module):
    def __init__(self,n_position, d_hid,drop_rate=0.2):
        super().__init__()
        self.n_position = n_position
        self.d_hid = d_hid
        self.pos_embed = nn.Parameter(torch.zeros(1, n_position,d_hid))

    def forward(self, x):
        x = x+self.pos_embed
        return x

class position_embedding(nn.Module):
    def __init__(self,n_position, d_hid,pos_type="learned",drop_rate=0.2):
        super().__init__()
        self.n_position = n_position
        self.d_hid = d_hid
        if pos_type == "learned":
            self.pos_embed = learned_position_enc(n_position, d_hid)
        elif pos_type == "cossin":
            self.pos_embed = position_enc(n_position, d_hid)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(self, x):
        return self.pos_embed(x)


# class transformers(nn.Module):
#     def __init__(self,num_layers,dim, in_dim, num_heads, mlp_ratio=1):
#         super().__init__()
#         self.num_layer = num_layers
#         self.layers = nn.ModuleList()
#         for i in range(num_layers-1):
#             block = transformer_bloack(dim,dim,num_heads,mlp_ratio,)
#             self.layers.append(copy.deepcopy(block))
#         self.last_layer = transformer_bloack(dim,in_dim,num_heads,mlp_ratio,)
#
#     def forward(self,x ):
#         if self.num_layer > 1 :
#             for block in self.layers:
#                 x = block(x)
#
#         x = self.last_layer(x)
#         return x
