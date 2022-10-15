import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, to_2tuple

from model.nn import Conv2d, Linear

def calc_dropout(dropout, sample_embed_dim, embed_dim):
    return dropout * sample_embed_dim / embed_dim

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, scale=False):
        super(PatchEmbed, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.embed_dim = embed_dim
        self.scale = scale

        self.sample_embed_dim = embed_dim

    def set_sample_config(self, embed_dim=None):
        self.sampled_embed_dim = embed_dim or self.sampled_embed_dim
        self.proj.set_sample_config(self.sampled_embed_dim, self.sampled_embed_dim)
        if self.scale:
            self.sampled_scale = self.embed_dim / self.sampled_embed_dim

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if self.scale:
            return x * self.sampled_scale
        return x

    def get_params(self):
        return self.proj.get_params()


class ClsToken(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        self.sample_embed_dim = embed_dim

    def get_params(self):
        return self.sample_embed_dim

    def set_sample_config(self, embed_dim=None):
        self.sample_embed_dim = embed_dim or self.sample_embed_dim

    def forward(self, x):
        N = x.shape[0]
        cls_token = self.cls_token[..., :self.sample_embed_dim].expand(N, -1, -1)
        return torch.cat((cls_token, x), dim=1)

class AbsPosEmbed(nn.Module):
    def __init__(self, length: int, embed_dim: int):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.sample_embed_dim = embed_dim

    def get_params(self):
        return self.length * self.sample_embed_dim

    def set_sample_config(self, embed_dim=None):
        self.sample_embed_dim = embed_dim or self.sample_embed_dim

    def forward(self, x):
        return x + self.pos_embed[..., :self.sample_embed_dim]



class RelativePosition2D(nn.Module):
    def __init__(self, head_dim, length=14):
        super(RelativePosition2D, self).__init__()

        self.head_dim = head_dim
        self.length = length
        # The first element in embeddings_table_v is the vertical embedding for the class
        self.embeddings_table_v = nn.Parameter(torch.randn(length * 2 + 2, head_dim))
        self.embeddings_table_h = nn.Parameter(torch.randn(length * 2 + 2, head_dim))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)

        self.sample_head_dim = head_dim

    def set_sample_config(self, head_dim=None):
        self.sample_head_dim = head_dim or self.sample_head_dim

    def get_params(self):
        return 2 * (self.length * 2 + 2) * self.sample_head_dim

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        device = self.embeddings_table_v.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        # compute the row and column distance
        sqrt_length = int(length_q ** 0.5 )
        # distance_mat_v = (range_vec_k[None, :] // int(length_q ** 0.5 )  - range_vec_q[:, None] // int(length_q ** 0.5 ))
        distance_mat_v = torch.div(range_vec_k[None, :], sqrt_length, rounding_mode='trunc') - torch.div(range_vec_q[:, None], sqrt_length, rounding_mode='trunc')
        distance_mat_h = (range_vec_k[None, :] % sqrt_length - range_vec_q[:, None] % sqrt_length)
        # clip the distance to the range of [-max_relative_position, max_relative_position]
        distance_mat_clipped_v = torch.clamp(distance_mat_v, -self.length, self.length)
        distance_mat_clipped_h = torch.clamp(distance_mat_h, -self.length, self.length)

        # translate the distance from [1, 2 * max_relative_position + 1], 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.length + 1
        final_mat_h = distance_mat_clipped_h + self.length + 1
        # pad the 0 which represent the cls token
        final_mat_v = F.pad(final_mat_v, (1, 0, 1, 0), "constant", 0)
        final_mat_h = F.pad(final_mat_h, (1, 0, 1, 0), "constant", 0)

        final_mat_v = final_mat_v.long()
        final_mat_h = final_mat_h.long()
        # get the embeddings with the corresponding distance
        embeddings = self.embeddings_table_v[final_mat_v, :self.sample_head_dim] + self.embeddings_table_h[final_mat_h, :self.sample_head_dim]

        return embeddings

class Attention(nn.Module):
    def __init__(self, 
        embed_dim, num_heads, head_dim=64,
        qkv_bias=False, qk_scale=None, 
        attn_drop=0., proj_drop=0., 
        relative_position = False,
        max_relative_position=14, 
        scale=False
    ):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or embed_dim // num_heads
        head_dim = self.head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.fc_scale = scale

        self.q = Linear(embed_dim, head_dim * num_heads, bias = qkv_bias)
        self.k = Linear(embed_dim, head_dim * num_heads, bias = qkv_bias)
        self.v = Linear(embed_dim, head_dim * num_heads, bias = qkv_bias)

        self.relative_position = relative_position
        if relative_position:
            self.rel_pos_embed_k = RelativePosition2D(head_dim, max_relative_position)
            self.rel_pos_embed_v = RelativePosition2D(head_dim, max_relative_position)
        self.max_relative_position = max_relative_position

        self.proj = Linear(head_dim * num_heads, embed_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sample_embed_dim = embed_dim
        self.sample_num_heads = num_heads


    def set_sample_config(self, embed_dim=None, num_heads=None):
        self.sample_embed_dim = embed_dim or self.sample_embed_dim
        self.sample_num_heads = num_heads or self.sample_num_heads

        self.q.set_sample_config(self.sample_embed_dim, self.sample_num_heads * self.head_dim)
        self.k.set_sample_config(self.sample_embed_dim, self.sample_num_heads * self.head_dim)
        self.v.set_sample_config(self.sample_embed_dim, self.sample_num_heads * self.head_dim)
        self.proj.set_sample_config(self.sample_num_heads * self.head_dim, self.sample_embed_dim)
        # if self.relative_position:
        #     self.rel_pos_embed_k.set_sample_config(self.head_dim)
        #     self.rel_pos_embed_v.set_sample_config(self.head_dim)

    def get_params(self):
        params = self.q.get_params() + self.k.get_params() + self.v.get_params() + self.proj.get_params()
        if self.relative_position:
            params += self.rel_pos_embed_k.get_params() + self.rel_pos_embed_v.get_params()
        return params

    def forward(self, x):
        B, N, _ = x.shape
        head_dim = self.head_dim
        num_heads = self.sample_num_heads
        q = self.q(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.sample_num_heads * B, -1) @ r_p_k.transpose(2, 1)) \
                .transpose(1, 0).reshape(B, self.sample_num_heads, N, N) * self.scale

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, -1)
        if self.relative_position:
            r_p_v = self.rel_pos_embed_v(N, N)
            attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * self.sample_num_heads, -1)
            # The size of attention is (B, num_heads, N, N), reshape it to (N, B*num_heads, N) and do batch matmul with
            # the relative position embedding of V (N, N, head_dim) get shape like (N, B*num_heads, head_dim). We reshape it to the
            # same size as x (B, num_heads, N, hidden_dim)
            x = x + (attn_1 @ r_p_v).transpose(1, 0).reshape(B, self.sample_num_heads, N, -1).transpose(2,1).reshape(B, N, -1)

        if self.fc_scale:
            x = x * (self.embed_dim / self.sample_embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == "__main__":
    model = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768, scale=False)
    inputs = torch.rand(1, 3, 224, 224)
    loss = model(inputs).mean()
    loss.backward()
    print(model.get_params())
    model.set_sample_config(192)
    print(model.get_params())