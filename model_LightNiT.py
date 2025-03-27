#-*-codeing = utf-8 -*-
#@Time : 2024/3/13 14:10
#@Author :ypthon
#@File : lightNiT.py
#@Software:PyCharm

from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import DropPath

class ConvStem1D(nn.Module):
    def __init__(self, seq_len=200, in_chans=6, embed_dim=128):
        super().__init__()
        self.seq_len = seq_len

        stem_dim = embed_dim * 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, stem_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(stem_dim),
            nn.ReLU(),
            nn.Conv1d(stem_dim, stem_dim, kernel_size=3, groups=stem_dim, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(stem_dim),
            nn.ReLU(),
            nn.Conv1d(stem_dim, stem_dim, kernel_size=3, groups=stem_dim, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(stem_dim),
            nn.ReLU(),
            nn.Conv1d(stem_dim, stem_dim, kernel_size=3, groups=stem_dim, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(stem_dim),
            nn.ReLU(),
            nn.Conv1d(stem_dim, stem_dim, kernel_size=3, groups=stem_dim, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(stem_dim),
            nn.ReLU(),
            nn.Conv1d(stem_dim, stem_dim, kernel_size=3, groups=stem_dim, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(stem_dim),
            nn.ReLU()

        )
        self.proj = nn.Conv1d(stem_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        stem = self.stem(x)
        x = self.proj(stem)
        _, _, L = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, L



class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.25, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn * s_attn  # [B, N, C]
        return ori_x * attn

class BiAttnMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.attn = BiAttn(out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.attn(x)
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
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=6., qkv_bias=False, drop=0.3, attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = BiAttnMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LightViT1D_MultiInput(nn.Module):
    def __init__(self, seq_len=200, embed_dim=128, num_heads=8, mlp_ratio=5., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., norm_layer=None, act_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.vel_patch_embed = ConvStem1D(seq_len=seq_len, in_chans=6, embed_dim=embed_dim)
        self.vel_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, act_layer=act_layer
            )
            for _ in range(4)
        ])
        self.vel_norm = norm_layer(embed_dim)
        self.displacement_head = nn.Linear(embed_dim, 3)  
        self.covariance_head = nn.Linear(embed_dim, 3)  
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, motion_input):

        x, _ = self.vel_patch_embed(motion_input)
        x = self.vel_blocks(x)
        x = self.vel_norm(x)
        x = x.mean(dim=1)  
        displacement = self.displacement_head(x)  # (batch_size, 3)
        covariance = self.covariance_head(x)  # (batch_size, 3)

        return displacement, covariance

class LightViT1D_MultiInput_S(nn.Module):
    def __init__(self, seq_len=200, embed_dim=128, num_heads=8, mlp_ratio=5., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., norm_layer=None, act_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.vel_patch_embed = ConvStem1D(seq_len=seq_len, in_chans=6, embed_dim=embed_dim)
        self.vel_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, act_layer=act_layer
            )
            for _ in range(1)
        ])
        self.vel_norm = norm_layer(embed_dim)
        self.displacement_head = nn.Linear(embed_dim, 3)  
        self.covariance_head = nn.Linear(embed_dim, 3)  
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, motion_input):

        x, _ = self.vel_patch_embed(motion_input)
        x = self.vel_blocks(x)
        x = self.vel_norm(x)
        x = x.mean(dim=1)  
        displacement = self.displacement_head(x)  # (batch_size, 3)
        covariance = self.covariance_head(x)  # (batch_size, 3)

        return displacement, covariance

class LightViT1D_MultiInput_R(nn.Module):
    def __init__(self, seq_len=200, embed_dim=128, num_heads=8, mlp_ratio=5., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., norm_layer=None, act_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.vel_patch_embed = ConvStem1D(seq_len=seq_len, in_chans=6, embed_dim=embed_dim)
        self.vel_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, act_layer=act_layer
            )
            for _ in range(2)
        ])
        self.vel_norm = norm_layer(embed_dim)
        self.displacement_head = nn.Linear(embed_dim, 3)  
        self.covariance_head = nn.Linear(embed_dim, 3)  
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, motion_input):

        x, _ = self.vel_patch_embed(motion_input)
        x = self.vel_blocks(x)
        x = self.vel_norm(x)
        x = x.mean(dim=1)  
        displacement = self.displacement_head(x)  # (batch_size, 3)
        covariance = self.covariance_head(x)  # (batch_size, 3)

        return displacement, covariance
