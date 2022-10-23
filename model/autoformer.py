import math

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.nn import LayerNorm, Linear, Identity
    from model.module import PatchEmbed, ClsToken, AbsPosEmbed, Attention
except:
    from nn import LayerNorm, Linear, Identity
    from module import PatchEmbed, ClsToken, AbsPosEmbed, Attention

from timm.models.layers import DropPath, trunc_normal_


def calc_dropout(dropout, sample_embed_dim, embed_dim):
    return dropout * sample_embed_dim / embed_dim


class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
        embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
        dropout=0., attn_dropout=0., drop_path=0., 
        act_layer=nn.GELU, pre_norm=True, 
        scale=False, grad_scale=True,
        relative_position=False, max_relative_position=14
    ):
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.intermediate_dim = int(mlp_ratio * embed_dim)
        self.num_heads = num_heads
        
        self.normalize_before = pre_norm
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position

        self.attn = Attention(
            embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_dropout,
            proj_drop=dropout, scale=self.scale, grad_scale=grad_scale,
            relative_position=self.relative_position, max_relative_position=max_relative_position
        )

        self.attn_layer_norm = LayerNorm(embed_dim, grad_scale=grad_scale)
        self.ffn_layer_norm = LayerNorm(embed_dim, grad_scale=grad_scale)

        self.act_layer = act_layer()

        self.fc1 = Linear(self.embed_dim, self.intermediate_dim, bias=True, grad_scale=grad_scale)
        self.fc2 = Linear(self.intermediate_dim, self.embed_dim, bias=True, grad_scale=grad_scale)

        # the configs of current sampled arch
        self.sample_embed_dim = embed_dim
        self.sample_mlp_ratio = mlp_ratio
        self.sample_intermediate_dim = int(mlp_ratio * embed_dim)
        self.sample_num_heads = num_heads

        self.sample_dropout = dropout
        self.sample_attn_dropout = attn_dropout

    def get_params(self):
        params = 0
        params += self.attn.get_params()
        params += self.attn_layer_norm.get_params()
        params += self.ffn_layer_norm.get_params()
        params += self.fc1.get_params()
        params += self.fc2.get_params()
        return params
    
    def set_sample_config(self, embed_dim=None, mlp_ratio=None, num_heads=None):
        self.sample_embed_dim = embed_dim or self.sample_embed_dim
        self.sample_mlp_ratio = mlp_ratio or self.sample_mlp_ratio
        self.sample_num_heads = num_heads or self.sample_num_heads
        self.sample_intermediate_dim = int(self.sample_embed_dim * self.sample_mlp_ratio)

        self.sample_dropout = calc_dropout(self.dropout, self.sample_embed_dim, self.embed_dim)
        self.sample_attn_dropout = calc_dropout(self.attn_dropout, self.sample_embed_dim, self.embed_dim)

        self.attn_layer_norm.set_sample_config(self.sample_embed_dim)
        self.ffn_layer_norm.set_sample_config(self.sample_embed_dim)

        self.attn.set_sample_config(self.sample_embed_dim, self.sample_num_heads)

        self.fc1.set_sample_config(self.sample_embed_dim, self.sample_intermediate_dim)
        self.fc2.set_sample_config(self.sample_intermediate_dim, self.sample_embed_dim)

    def forward(self, x):
        # compute attn
        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
        # compute the ffn
        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.act_layer(self.fc1(x))
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.scale:
            x = x * (self.mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x



class VisionTransformer(nn.Module):
    def __init__(self, 
        embed_dim = 768, 
        depth = 12,
        num_heads = 12, 
        mlp_ratio = 4., 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_classes=1000, 
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0.,
        drop_path_rate=0., 
        pre_norm=True, 
        scale=False, 
        grad_scale=False,
        global_pool=False, 
        relative_position=False, 
        abs_pos = True, 
        max_relative_position=14, **kwargs
    ):
        super(VisionTransformer, self).__init__()

        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.num_heads = num_heads
        
        self.num_classes = num_classes
        self.dropout = drop_rate
        self.attn_dropout = attn_drop_rate
        self.pre_norm = pre_norm
        self.scale = scale

        self.global_pool = global_pool
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, grad_scale=grad_scale
        )
        # parameters for vision transformer
        num_patches = self.patch_embed.num_patches
        
        # stochastic depth decay rule
        layers = []
        for dpr in np.linspace(0, drop_path_rate, depth):
            layers.append(
                TransformerEncoderLayer(
                    embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    dropout=drop_rate, attn_dropout=attn_drop_rate, drop_path=dpr,
                    pre_norm=pre_norm, scale=self.scale, grad_scale=grad_scale,
                    relative_position=relative_position, max_relative_position=max_relative_position
                )
            )
        self.layers = nn.ModuleList(layers)

        self.cls_token = ClsToken(embed_dim)
        self.pos_embed = AbsPosEmbed(num_patches + 1, embed_dim) if abs_pos else Identity()

        self.norm = LayerNorm(embed_dim, grad_scale=grad_scale) if pre_norm else Identity()

        # classifier head
        self.classifier = Linear(embed_dim, num_classes, bias=True, grad_scale=grad_scale) if num_classes > 0 else Identity()

        self.apply(self._init_weights)

        # configs for the sampled subTransformer
        self.sample_embed_dim = embed_dim
        self.sample_depth = depth
        self.sample_num_heads = num_heads
        self.sample_mlp_ratio = mlp_ratio
        self.sample_dropout = drop_rate

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        else:
            pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_embed'}

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = min(config['embed_dim'], self.embed_dim)
        self.sample_depth = min(config['depth'], self.depth)
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_num_heads = config['num_heads']

        self.sample_dropout = calc_dropout(self.dropout, self.sample_embed_dim, self.embed_dim)

        self.patch_embed.set_sample_config(self.sample_embed_dim)
        self.cls_token.set_sample_config(self.sample_embed_dim)
        self.pos_embed.set_sample_config(self.sample_embed_dim)

        # not exceed sample layer number
        for i in range(self.sample_depth):
            layer = self.layers[i]
            layer.set_sample_config(self.sample_embed_dim, self.sample_mlp_ratio[i], self.sample_num_heads[i])

        self.norm.set_sample_config(self.sample_embed_dim)
        self.classifier.set_sample_config(self.sample_embed_dim, self.num_classes)

    def get_params(self):
        params = 0
        params += self.patch_embed.get_params()
        for i in range(self.sample_depth):
            params += self.layers[i].get_params()
        params += self.cls_token.get_params()
        params += self.pos_embed.get_params()
        params += self.norm.get_params()
        params += self.classifier.get_params()
        return params

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cls_token(x)
        x = self.pos_embed(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        # x = self.layers(x)
        for i in range(self.sample_depth):
            x = self.layers[i](x)
        x = self.norm(x)

        if self.global_pool:
            x = torch.mean(x[:, 1:], dim=1)
        else:
            x = x[:, 0]
        x = self.classifier(x)
        return x




if __name__ == "__main__":
    import random
    from itertools import product
    search_space = {
        "embed_dim": [192, 216, 240],
        "depth": [12, 13, 14],
        "num_heads": [3, 4],
        "mlp_ratio": [3.0, 3.5, 4.0]
    }
    config = {
        "embed_dim": random.choice(search_space["embed_dim"]),
        "depth": random.choice(search_space["depth"]),
        "num_heads": [random.choice(search_space["num_heads"]) for _ in range(max(search_space["depth"]))],
        "mlp_ratio": [random.choice(search_space["mlp_ratio"]) for _ in range(max(search_space["depth"]))]
    }
    config['embed_dim'] = 192
    config['depth'] = 12
    # print(config)
    model = VisionTransformer(
        img_size=224, patch_size=16, num_classes=1000,
        embed_dim=240, depth=14, num_heads=4, mlp_ratio=4.0, qkv_bias=True, 
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, global_pool=True,
        max_relative_position=14, relative_position=True, abs_pos=True, grad_scale=True
    )

    def mask1d(chans):
        m = torch.arange(len(chans), 0, -1)
        m = torch.repeat_interleave(m, torch.diff(torch.tensor([0] + chans)), dim=0)
        return 1.0 / m

    def mask2d(in_chans, out_chans):
        m = torch.arange(len(out_chans), 0, -1).reshape(-1, 1) * torch.arange(len(in_chans), 0, -1).reshape(1, -1)

        m = torch.repeat_interleave(m, torch.diff(torch.tensor([0] + out_chans)), dim=0)
        m = torch.repeat_interleave(m, torch.diff(torch.tensor([0] + in_chans)), dim=1)
        return 1.0 / m

    def gen_mask(model, search_space):
        state_dict = model.state_dict()
        for i, d in enumerate(sorted(search_space['embed_dim'], reverse=True), start=1):
            state_dict['patch_embed.proj.mask_weight'][:d] = 1.0/i
            state_dict['patch_embed.proj.mask_bias'][:d] = 1.0/i
        
        for l in range(14):
            for t in ['q', 'k', 'v']:
                state_dict[f'layers.{l}.attn.{t}.mask_weight'] = mask2d(search_space['embed_dim'], [64*i for i in search_space['num_heads']])
                state_dict[f'layers.{l}.attn.{t}.mask_bias'] = mask1d([64*i for i in search_space['num_heads']])
            state_dict[f'layers.{l}.attn.proj.mask_weight'] = mask2d([64*i for i in search_space['num_heads']], search_space['embed_dim'])
            state_dict[f'layers.{l}.attn.proj.mask_bias'] = mask1d(search_space['embed_dim'])

            for t in ['attn_layer_norm', 'ffn_layer_norm']:
                state_dict[f'layers.{l}.{t}.mask_weight'] = mask1d(search_space['embed_dim'])
                state_dict[f'layers.{l}.{t}.mask_bias'] = mask1d(search_space['embed_dim'])

            # intermediate_dim = [int(i*j) for i, j in product(search_space['embed_dim'], search_space['mlp_ratio'])]
            # intermediate_dim = sorted(intermediate_dim)
            # intermediate_dim = [int(i*216) for i in search_space['mlp_ratio']]
            # state_dict[f'layers.{l}.fc1.mask_weight'] = mask2d(search_space['embed_dim'], intermediate_dim)
            # state_dict[f'layers.{l}.fc1.mask_bias'] = mask1d(intermediate_dim)
            # state_dict[f'layers.{l}.fc2.mask_weight'] = mask2d(intermediate_dim, search_space['embed_dim'])
            # state_dict[f'layers.{l}.fc2.mask_bias'] = mask1d(search_space['embed_dim'])
        state_dict[f'norm.mask_weight'] = mask1d(search_space['embed_dim'])
        state_dict[f'norm.mask_bias'] = mask1d(search_space['embed_dim'])

        state_dict[f'classifier.mask_weight'] = mask2d(search_space['embed_dim'], [1000])
        state_dict[f'classifier.mask_bias'] = mask1d([1000])
        return state_dict

    # for k, v in model.state_dict().items():
    #     print(k, tuple(v.shape))
    model.load_state_dict(gen_mask(model, search_space))
    
    inputs = torch.rand(8, 3, 224, 224)
    target = torch.randint(0, 3, (8, ))
    with torch.no_grad():
        logits = model(inputs)
        print(model.get_params())
        model.set_sample_config(config)
        logits = model(inputs)
        print(model.get_params())
    
    F.cross_entropy(model(inputs), target).backward()
    