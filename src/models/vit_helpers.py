"""Main blocks used for VIT and other architectures.

We incorporate code from mmAction2 (https://github.com/open-mmlab/mmaction2) and AVION (
https://github.com/zhaoyue-zephyrus/AVION).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import drop_path, to_2tuple, trunc_normal_
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2

import xformers.ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

import ipdb
from .static_token_utils import batched_find_idxs_to_keep



class DropPath(nn.Module):
    """Implements stochastic depth.

    Taken from
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers.py
    """

    def __init__(self, drop_prob: float = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    """FFN for transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # TODO: update this to the more standard TIMM
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """MHA implementation for VIT."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = 1024,
        use_flash_attn: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        #head_dim = 85
        all_head_dim = head_dim * self.num_heads
        #ipdb.set_trace()
        #all_head_dim = 1020
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        #self.qkv = nn.Linear(dim, 1020, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_flash_attn = use_flash_attn

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Scaled DP attention."""
        B, N, C = x.shape
        qkv_bias = None
        # AVION uses a QKV bias. We'll stick with this.
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, 
                 torch.zeros_like(self.v_bias, requires_grad=False), 
                 self.v_bias)
            )

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        if self.use_flash_attn:
            # TODO: ensure this is the most optimal impl. 
            # Also, should encompass the linear + proj steps in a single kernel.
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = xops.fmha.memory_efficient_attention(q, k, v, 
                                                     p=self.attn_drop_rate, 
                                                     scale=self.scale, 
                                                     attn_bias=attn_mask)
        else:
            # Non-optimized version.
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2)


        x = x.reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        use_flash_attn=True,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            attn_head_dim=attn_head_dim,
            use_flash_attn=use_flash_attn
        )
        
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_rate
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mask=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), attn_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), attn_mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_frames: int = 16,
        tubelet_size: int = 2,
        channels_last: bool = False,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        # N_patches = 224/16 * 224/16 * 16/2 = 1568
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames // self.tubelet_size)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.channels_last = channels_last
        self.embed_dim = embed_dim
        if channels_last:
            self.proj = nn.Linear(
                in_features=in_channels * tubelet_size * patch_size[0] * patch_size[1],
                out_features=embed_dim,
            )
        else:
            self.proj = nn.Conv3d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                stride=(self.tubelet_size, patch_size[0], patch_size[1]),
            )

    def forward(self, x: Tensor, token_mask: Tensor=None) -> Tensor:
        if self.channels_last:
            x = rearrange(
                x,
                "b c (t p0) (h p1) (w p2) -> b (t h w) (c p0 p1 p2)",
                p0=self.tubelet_size,
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            )
            x = self.proj(x)
            return x
        else:
            # Rearranging / tokenization moved to CPU.
            n_toks, C, T, H, W = x.shape

            if token_mask is not None:
                x = x[token_mask]

            x = self.proj(x).reshape(1, n_toks, self.embed_dim)
            
            return x


def get_sinusoid_encoding(n_position: int, embed_dims: int) -> Tensor:
    """Generate sinusoid encoding table.

    Sinusoid encoding is a kind of relative position encoding method came from
    Args:
        n_position (int): The length of the input token.
        embed_dims (int): The position embedding dimension.
    Returns:
        `torch.FloatTensor`: The sinusoid encoding table of size
        (1, n_position, embed_dims)
    """

    vec = torch.arange(embed_dims, dtype=torch.float64)
    vec = (vec - vec % 2) / embed_dims
    vec = torch.pow(10000, -vec).view(1, -1)

    sinusoid_table = torch.arange(n_position).view(-1, 1) * vec
    sinusoid_table[:, 0::2].sin_()  # dim 2i
    sinusoid_table[:, 1::2].cos_()  # dim 2i+1

    sinusoid_table = sinusoid_table.to(torch.float32)

    return sinusoid_table.unsqueeze(0)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 400,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        fc_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: float = 0.0,
        use_learnable_pos_emb: bool = False,
        init_scale: float = 0.001,
        num_frames: int = 16,
        tubelet_size: int = 2,
        channels_last: bool = False,
        use_flash_attn: bool = False,
        use_mean_pooling: bool = True,
        use_length_embed: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dims,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
            channels_last=channels_last,
        )
        # Add configs; this is now part of the "tokenizer config i suppose"
        
        self.in_channels = in_channels
        grid_size = img_size // patch_size
        num_patches = grid_size**2 * (num_frames // tubelet_size)
        self.grid_size = (grid_size, grid_size)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding(num_patches, embed_dims)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.use_length_embed = use_length_embed
        if use_length_embed:
            self.length_embed = nn.Parameter(torch.zeros(1, num_frames // tubelet_size, embed_dims), requires_grad=True)
            nn.init.trunc_normal_(self.length_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    use_flash_attn=use_flash_attn,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dims)
        self.fc_norm = norm_layer(embed_dims) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: Tensor, 
                         batch_size: int,
                         attn_mask: Optional[Tensor] = None, 
                         pos_embed: Optional[Tensor] = None, 
                         length_embed: Optional[Tensor] = None) -> Tensor:
        # X = (B*n_crops, n_tokens, C, T, H, W)
        x = self.patch_embed(x)

        if attn_mask is None:
            # pos embed is not pre-computed, and is on CPU.
            self.pos_embed = self.pos_embed.to(x)
            pos_embed = self.pos_embed
        x = x + pos_embed.detach()
        # Add the optional length embedding.
        if length_embed is not None:
            x = x + length_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.norm(x)

        if self.fc_norm is not None:            
            # Want to find class per block; split with attn_mask block sizes.
            #ipdb.set_trace()
            splits = [seq.mean(1) for seq in attn_mask.split(x)]
            # Split all the sequences, so we have B, n_crops, embed_dim
            #ipdb.set_trace()
            x = torch.vstack(splits)
            x = x.reshape(batch_size, -1, self.embed_dim)
            return self.fc_norm(x)

        return x[:, 0]

    def forward(self, x: Tensor, 
                num_tokens: Tensor, 
                pos_embed: Tensor, 
                batch_size: int,
                lengths: Tensor = None) -> Tensor:
        #ipdb.set_trace()
        pos_embed = pos_embed.unsqueeze(0)
        
        num_tokens = num_tokens.flatten().type(torch.int).tolist()
        attn_mask = BlockDiagonalMask.from_seqlens(num_tokens)
        length_embeds = None
        if self.use_length_embed:
            length_embeds = torch.index_select(self.length_embed, 1, lengths - 1)
        x = self.forward_features(x, batch_size=batch_size, attn_mask=attn_mask, pos_embed=pos_embed, length_embed=length_embeds)
        x = self.fc_dropout(x)
        x = self.head(x)
        # remove the "clip/crop" dimension by avg across it.
        return x