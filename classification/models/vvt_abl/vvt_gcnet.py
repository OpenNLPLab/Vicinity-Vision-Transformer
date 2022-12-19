import math
from functools import partial
from sre_constants import AT_NON_BOUNDARY
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from torch import Tensor
from utils.utils import logging_info


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x, H, W):
        B, N, C=x.shape
        x = x.view(B,C,H,W)
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        
        out = out.view(B,N,C)

        return out





class Vanilla_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class pvt_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        logging_info(f"linear: {linear}, sr_ratio: {sr_ratio}")

        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)
    
    def get_attn(self):
        return self.attn_map

    def save_attn(self, attn):
        self.attn_map = attn

    def save_attn_gradients(self, attn_gradients):
        self.attn_map_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_map_gradients

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, h, N, C/h

        if not self.linear: 
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # save the attention map for visualization
        if x.requires_grad:
            self.save_attn(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class VicinityVisionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None,
                 dropout_rate=0.0, causal=False, use_sum=True, 
                 sr_ratio=1, fr_ratio=1, linear=False, se_reduction=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
		# q, k, v projection
        self.fr = fr_ratio 
        # feature reduction
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.fr)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.fr)
        self.q_proj = nn.Linear(embed_dim, embed_dim // self.fr)

        self.sr_ratio = sr_ratio
        self.linear = linear
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(embed_dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(embed_dim)
            self.act = nn.GELU()

        self.apply(self._init_weights)
        # outprojection
        self.out_proj = nn.Linear(embed_dim//self.fr, embed_dim)
		# dropout rate
        self.dropout_rate = dropout_rate
		# causal
        self.causal = causal

        assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"
        self.use_sum = use_sum
        if self.use_sum:
            print('use sum')
            logging_info(f"linear: {linear}, sr_ratio: {sr_ratio}, fr_ratio: {fr_ratio} se_ratio: {se_reduction}")
        else:
            print('use production')
            logging_info(f"linear: {linear}, sr_ratio: {sr_ratio}, fr_ratio: {fr_ratio} se_ratio: {se_reduction}")

        # se block:
        reduction = se_reduction
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // reduction, embed_dim, bias=False),
            nn.Sigmoid()
        )

        self.clip = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def abs_clamp(self, t):
        min_mag = 1e-4
        max_mag = 10000
        sign = t.sign()
        return t.abs_().clamp_(min_mag, max_mag)*sign
        
    def get_index(self, m, n):
        """
        m = width, n = height
        """
        c = np.pi / 2
        seq_len = m * n
        index = torch.arange(seq_len).reshape(1, -1, 1, 1)
        a = c * (index // m) / n
        b = c * (index % m) / m

        seq_len = (m/self.sr_ratio) * (n/self.sr_ratio)
        index = torch.arange(seq_len).reshape(1, -1, 1, 1)
        a_sr = c * (index // (m/self.sr_ratio) ) / (n/self.sr_ratio)
        b_sr = c * (index % (m/self.sr_ratio)) / (m/self.sr_ratio)

        return nn.Parameter(a, requires_grad=False), nn.Parameter(b, requires_grad=False), \
               nn.Parameter(a_sr, requires_grad=False), nn.Parameter(b_sr, requires_grad=False)

    
    def abs_clamp(self, t):
        min_mag = 1e-4
        max_mag = 10000
        sign = t.sign()
        return t.abs_().clamp_(min_mag, max_mag)*sign

    def forward(self, query, H, W):
        # H: height, W: weight
        num_heads = self.num_heads
        B, N, C = query.shape
        query_se = query.permute(0, 2, 1)
        query_se = self.se_pool(query_se).view(B, C)
        query_se = self.se_fc(query_se).view(B, C, 1)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = query.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                k = self.k_proj(x_)
                v = self.v_proj(x_)
            else:
                k = self.k_proj(query)
                v = self.v_proj(query)
        else:
            x_ = query.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            k = self.k_proj(x_)
            v = self.v_proj(x_)

        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        query = query.permute(1,0,2)
        tgt_len, bsz, embed_dim = query.size()
        head_dim = embed_dim // num_heads
        # (L, N, E)
        q = self.q_proj(query)
        
		# multihead
		# (N, L, h, d)
        q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim//self.fr).transpose(0, 1)
		# (N, S, h, d)
        k = k.contiguous().view(-1, bsz, num_heads, head_dim//self.fr).transpose(0, 1)
		# (N, S, h, d)
        v = v.contiguous().view(-1, bsz, num_heads, head_dim//self.fr).transpose(0, 1)
		# relu
        q = F.relu(q)
        k = F.relu(k)

        a, b, a_sr, b_sr = self.get_index(W, H)
        a = a.to(q)
        b = b.to(q)
        a_sr = a_sr.to(q)
        b_sr = b_sr.to(q)

        if self.use_sum:
            # sum
            q_ = torch.cat([q * torch.cos(a), \
                            q * torch.sin(a), \
                            q * torch.cos(b), \
                            q * torch.sin(b)], \
                            dim=-1)
            # (N, S, h, 4 * d)
            k_ = torch.cat([k * torch.cos(a_sr), \
                            k * torch.sin(a_sr), \
                            k * torch.cos(b_sr), \
                            k * torch.sin(b_sr)], \
                            dim=-1)
        else:
            q_ = torch.cat([q * torch.cos(a) * torch.cos(b), \
                            q * torch.cos(a) * torch.sin(b), \
                            q * torch.sin(a) * torch.cos(b), \
                            q * torch.sin(a) * torch.sin(b)], \
                            dim=-1)
            # (N, S, h, 4 * d)
            k_ = torch.cat([k * torch.cos(a_sr) * torch.cos(b_sr), \
                            k * torch.cos(a_sr) * torch.sin(b_sr), \
                            k * torch.sin(a_sr) * torch.cos(b_sr), \
                            k * torch.sin(a_sr) * torch.sin(b_sr)], \
                            dim=-1)

        eps = 1e-4

        #---------------------------------------------------------------------------------
        kv_ = torch.matmul(k_.permute(0, 2, 3, 1), v.permute(0, 2, 1, 3))  # no einsum  
        if self.clip:
            kv_ = self.abs_clamp(kv_)
        #---------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------
        k_sum = torch.sum(k_, axis=1, keepdim=True) # no einsum                         
        z_ = 1 / (torch.sum(torch.mul(q_, k_sum), axis=-1) + eps) # no einsum           
        if self.clip:
            z_ = self.abs_clamp(z_)
        #--------------------------------------------------------------------------------

        # no einsum---------------------------------------------------------------------
        attn_output = torch.matmul(q_.transpose(1, 2), kv_).transpose(1, 2)
        if self.clip:
            attn_output = self.abs_clamp(attn_output)
        # nlhm,nlh -> nlhm
        attn_output = torch.mul(attn_output, z_.unsqueeze(-1))
        if self.clip:
            attn_output = self.abs_clamp(attn_output)
        #--------------------------------------------------------------------------------
        
        # (N, L, h, d) -> (L, N, h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim//self.fr)

        attn_output = self.out_proj(attn_output)
        if self.clip:
            attn_output = self.abs_clamp(attn_output)

        # ------------------------------------- se block
        attn_output = attn_output.permute(1,2,0)
        attn_output = attn_output + attn_output * query_se.expand_as(attn_output)
        if self.clip:
            attn_output = self.abs_clamp(attn_output)
        attn_output = attn_output.permute(2,0,1)
        # -------------------------------------------------

        attn_output = attn_output.permute(1,0,2)

        return attn_output

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 sr_ratio=1, fr_ratio = 1, linear=False, seq_len=3136, se_ratio=1, prod_type="right"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.seq_len = seq_len
        
        # self.attn = VicinityVisionAttention(embed_dim=dim, num_heads=num_heads, sr_ratio=sr_ratio, fr_ratio = fr_ratio, linear=linear, se_reduction=se_ratio)
        # self.attn = Vanilla_Attention(dim=dim, num_heads=num_heads)
        self.attn = ContextBlock(inplanes = dim,ratio = 1 )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class VicinityVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], fr_ratios = [1,2,2,4], num_stages=4, linear=False, se_ratio=1, prod_types=["left", "left", "left", "left"]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            
            if i == 0:
                seq_len = (img_size//patch_size)**2
            else:
                seq_len = ((img_size // (2 ** (i + 1))) // 2)**2

            logging_info(f"seq_len: {seq_len}, embed dim: {embed_dims[i]}, num heads: {num_heads[i]}, depth: {depths[i]}")

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i],fr_ratio = fr_ratios[i], linear=linear, seq_len=seq_len, se_ratio=se_ratio, prod_type=prod_types[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

### VVT
@register_model
def vvt_gcnet_test(pretrained=False, **kwargs):
    model = VicinityVisionTransformer(
        patch_size=4, embed_dims=[32, 32, 32, 32], num_heads=[1, 1, 1, 1], mlp_ratios=[1, 1, 1, 1], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 1, 1, 1], sr_ratios=[1, 1, 1, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def vvt_gcnet_tiny(pretrained=False, **kwargs):
    model = VicinityVisionTransformer(
        patch_size=4, embed_dims=[96, 160, 320, 512], num_heads=[1,2,5,8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], 
        sr_ratios=[1,1,1,1], fr_ratios=[2,2,2,2],se_ratio=1,
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def vvt_gcnet_small(pretrained=False, **kwargs):
    model = VicinityVisionTransformer(
        patch_size=4, embed_dims=[96, 160, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 9, 3], 
        sr_ratios=[1,1,1,1],fr_ratios=[2,2,2,2],se_ratio=1,
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def vvt_gcnet_medium(pretrained=False, **kwargs):
    model = VicinityVisionTransformer(
        patch_size=4, embed_dims=[96, 160, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 27, 3], 
        sr_ratios=[1,1,1,1],fr_ratios=[2,2,2,2],se_ratio=1,
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def vvt_gcnet_large(pretrained=False, **kwargs):
    model = VicinityVisionTransformer(
        patch_size=4, embed_dims=[96, 160, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[4, 4, 36, 4], 
        sr_ratios=[1,1,1,1],fr_ratios=[2,2,2,2],se_ratio=1,
        **kwargs)
    model.default_cfg = _cfg()

    return model