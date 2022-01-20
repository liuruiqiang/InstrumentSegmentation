import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from transformer_block import position_embedding
from token_transformer import Token_transformer
from einops import rearrange
import numpy as np


class Conv2dReLU1x1(nn.Module):
    """
    [Conv2d(in_channels, out_channels, kernel),
    BatchNorm2d(out_channels),
    ReLU,]
    """
    def __init__(self, in_channels, out_channels, kernel=1, padding=0, bn=False):
        super(Conv2dReLU1x1, self).__init__()
        modules = OrderedDict()
        modules['conv'] = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        if bn:
            modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['in'] = nn.InstanceNorm2d(out_channels)
        modules['relu'] = nn.ReLU(inplace=True)
        self.l = nn.Sequential(modules)

    def forward(self, x):
        x = self.l(x)
        return x

class MFFM(nn.Module):
    """
    Multi frame fusion module
    """
    def __init__(self,in_channel,mode='add'):
        super(MFFM, self).__init__()
        self.mode = mode
        if mode == 'add':
            self.fusion = Conv2dReLU1x1(in_channel,in_channel*3)
        else:
            self.fusion = Conv2dReLU1x1(in_channel*3,in_channel*3)


    def forward(self,x):
        """
        Args:
            x: b,c,h,w(t,c,h,w)
        Returns:
        """
        t,c,h,w = x.shape
        x = x.unsqueeze(0)
        if(self.mode == 'add'):
            x = torch.sum(x,dim=1)
        else:
            items = torch.unbind(x,dim=1)
            x = torch.stack(items,dim=1)
        x = self.fusion(x)
        x = x.view(t,c,h,w)
        return x

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 256]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 1024, 3, 1)
        elif level==1:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level==2:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

class TransformerBlock(nn.Module):
    def __init__(self,length,embedding,num_heads):
        super(TransformerBlock, self).__init__()
        self.pose_embed = position_embedding(length,embedding)
        self.transformer = Token_transformer(embedding,embedding,num_heads)

    def forward(self,x,t=3):
        b, c, h, w = x.size()
        # print('trans_shape:', trans_input.shape)
        trans_input = x.view(b//t, t, c, h, w)
        trans_input = rearrange(trans_input, 'b t c h w -> b (t h w) c')
        posed_trans_input = self.pose_embed(trans_input)
        spatial_out = self.transformer(posed_trans_input)
        trans_out = rearrange(spatial_out, 'b (t h w) c -> b t c h w', t=3, h=h, w=w)
        trans_out = trans_out.view(b, c, h, w)
        return trans_out

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim

        self.head_dim = in_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, in_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, in_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    # @get_local("attn_map")
    def forward(self, x):
        B, N, C = x.shape
        # print('x shape', x.shape)
        # self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        # self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)
        if self.sr_ratio > 1:
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim ).permute(0, 2, 1, 3)
            x_ = x.permute(0, 2, 1).reshape(B, C,int(np.sqrt(N)), int(np.sqrt(N)))
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # print(x_.shape)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim ).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn_ = attn.clone()
        attn_map = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_map)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.sr_ratio == 1:
            v = v.transpose(1,2).reshape(B, N, self.in_dim)

            x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x

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

class Multi_Gran_Transformer_improved(nn.Module):
    """
    先做不同尺度的unfold，然后做pool成不同尺度的特征进行堆积成为key和value
    """
    def __init__(self, dim, in_dim, num_heads,num_levels,input_resolution, keep_init=True,multi_gran_opt='add',mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, sr_ratio=1,act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.unfolds = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.keep_init = keep_init
        self.multi_gran_opt = multi_gran_opt
        self.norms = nn.ModuleList()
        self.folds = nn.ModuleList()
        self.h, self.w = input_resolution[0],input_resolution[1]
        # print('heads:...',in_dim,num_heads,type(in_dim),type(num_heads))
        if keep_init:
            self.init_attn = Attention(
                dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, sr_ratio=sr_ratio)
        self.num_levels = num_levels
        for i in range(1,num_levels):
            kerner_size = 2**i
            stride = 2**i
            self.unfolds.append(nn.Unfold(kernel_size=(kerner_size, kerner_size), stride=stride))
            self.folds += [
                nn.Fold(output_size=(input_resolution[0],input_resolution[1]),kernel_size=(kerner_size, kerner_size), stride=stride)
            ]
            self.attentions += [
                Attention(
                    dim*(stride**2), in_dim=in_dim*(stride**2), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    proj_drop=drop, sr_ratio=sr_ratio)
            ]
            self.norms += [
                norm_layer(dim*(stride**2))
            ]
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        if self.multi_gran_opt == 'cat' and self.keep_init:
            self.norm2 = norm_layer(in_dim*(num_levels))
            self.mlp = Mlp(in_features=in_dim*(num_levels), hidden_features=int(in_dim *(num_levels)* mlp_ratio),
                           out_features=in_dim*(num_levels),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(in_dim*(num_levels),in_dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(in_dim*(num_levels))
        elif self.multi_gran_opt == 'cat' and not self.keep_init:
            self.norm2 = norm_layer(in_dim * (num_levels-1))
            self.mlp = Mlp(in_features=in_dim * (num_levels-1), hidden_features=int(in_dim*(num_levels-1) * mlp_ratio),
                           out_features=in_dim*(num_levels-1),
                           act_layer=act_layer, drop=drop)
            self.mlp2 = nn.Linear(in_dim * (num_levels-1), in_dim)
            self.drop_path2 = nn.Dropout(0.1)
            self.norm3 = norm_layer(in_dim * (num_levels-1))
        else:
            self.norm2 = norm_layer(in_dim)
            self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def reshape_out(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, c)
        return x

    # @get_local("attention_tmp")
    def forward(self, x):
        # print(self.unfolds)
        x_tmp = x.clone()
        B, new_HW, C = x.shape
        x_tmp = x_tmp.transpose(1, 2).reshape(B, C, self.h, self.w)
        x_splits = []
        if self.num_levels >1:
            for i in range(1,self.num_levels):
                split_tmp = self.unfolds[i-1](x_tmp).permute(0,2,1)
                # print('split_tmp',split_tmp.shape)
                attntion_tmp = self.attentions[i-1](self.norms[i-1](split_tmp))
                attntion_tmp = attntion_tmp.permute(0,2,1)
                fold_tmp = self.folds[i-1](attntion_tmp)
                x_splits.append(self.reshape_out(fold_tmp))

        if self.keep_init:
            x = self.init_attn(self.norm1(x))
            attention_tmp = x
        if len(x_splits) > 0:
            if self.multi_gran_opt == 'add':
                cat_tmp = torch.stack(x_splits, dim=0)
                split_result = torch.sum(cat_tmp,dim=0)
                if not self.keep_init:
                    x = split_result
                else:
                    x += split_result

            elif self.multi_gran_opt == 'cat':
                split_cat = torch.cat(x_splits,dim=2)
                if not self.keep_init:
                    x = split_cat
                else:
                    x = torch.cat((x, split_cat), dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.multi_gran_opt == "cat":
            x = self.drop_path2(self.mlp2(self.norm3(x)))
        return x

class FocalTransformerBlock(nn.Module):
    def __init__(self,length,embedding,num_heads,num_levels=1,input_resolution=(10,8)):
        super(FocalTransformerBlock, self).__init__()
        self.pose_embed = position_embedding(length,embedding)
        self.transformer = Multi_Gran_Transformer_improved(embedding,embedding,num_heads,num_levels=num_levels,
                                                           input_resolution=input_resolution)

    def forward(self,x,t=3):
        b, c, h, w = x.size()
        # print('trans_shape:', trans_input.shape)
        trans_input = x.view(b,c, h, w)
        trans_input = rearrange(trans_input, 'b c h w -> b (h w) c')
        posed_trans_input = self.pose_embed(trans_input)
        spatial_out = self.transformer(posed_trans_input)
        trans_out = rearrange(spatial_out, 'b (h w) c -> b c h w', h=h, w=w)
        trans_out = trans_out.view(b, c, h, w)
        return trans_out

# data = torch.rand(2,512,10,8).cuda()
# model = FocalTransformerBlock(80,512,4,2).cuda()
# out = model(data)
# print(out.shape)