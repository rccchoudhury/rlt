"""Main blocks used for VIT and other architectures.

We incorporate code from mmAction2 (https://github.com/open-mmlab/mmaction2) and AVION (
https://github.com/zhaoyue-zephyrus/AVION).
"""

from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import drop_path, to_2tuple, trunc_normal_
from torch import Tensor
import numpy as np
import ipdb
from .tome_utils import bipartite_soft_matching, merge_source, merge_wavg, parse_r
from .vit_helpers import DropPath, Mlp, PatchEmbed, get_sinusoid_encoding


class ToMeAttention(nn.Module):
    """MHA implementation for VIT."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, size: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Scaled DP attention."""
        B, N, C = x.shape
        qkv_bias = None
        # Read abou
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )
        #import ipdb; ipdb.set_trace()
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Implements proportional attention.
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k.mean(1)


class ToMeBlock(nn.Module):
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
        use_flash_attn=False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        assert not use_flash_attn
        self.attn = ToMeAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            attn_head_dim=attn_head_dim,
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

    def forward(self, x: Tensor) -> Tensor:
        #assert self._tome_info["prop_attn"] and self._tome_info["r"] > 0
        assert self.gamma_1 is None
        attn_size = self._tome_info["size"]
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self.drop_path(x_attn)
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # print(self._tome_info)
        # Want it to be a list.
        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(merge, x, self._tome_info["source"])
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformerToMe(nn.Module):
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
        trace_source: bool = False,
        r: Union[List[int],  int] = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dims,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
            channels_last=channels_last,
        )
        self.r = r
        # TODO: MOVE TO CONFIG.
        self._tome_info = {
            "r": r,
            "size": None,  # start here...
            "source": None,
            "trace_source": trace_source,
            "prop_attn": True,
            # why would this be fals?
            "class_token": False,
            "distill_token": False,
        }

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

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                ToMeBlock(
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

        for module in self.blocks:
            module._tome_info = self._tome_info


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: Tensor, batch_size: int, pos_embed: Optional[Tensor] = None, num_tokens: Optional[Tensor] = None) -> Tensor:
        # X = (B*n_crops, n_tokens, C, T, H, W)
        # Re-enable this.
        #ipdb.set_trace()
        self._tome_info["size"] = None
        x = self.patch_embed(x, token_mask=None)
        n_clips = num_tokens.shape[0]
        x = x.reshape(n_clips, -1, self.embed_dims)
            # pos embed is not pre-computed, and is on CPU.
        x = x + pos_embed.detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        #ipdb.set_trace()
        x = self.norm(x)

        if self.fc_norm is not None:
            #ipdb.set_trace()
            #num_clips = num_tokens.shape[0] // batch_size
            #x = x.reshape(batch_size, num_clips, -1, self.embed_dims)
            x = x.mean(1)
            return self.fc_norm(x)

        return x[:, 0]
    
    def forward(self, x: Tensor, 
            num_tokens: Tensor, 
            pos_embed: Tensor, 
            batch_size: int,
            lengths: Tensor = None) -> Tensor:
        #ipdb.set_trace()
        n_clips = num_tokens.shape[0]
        pos_embed = pos_embed.reshape(n_clips, -1, self.embed_dims)
        
        self._tome_info["r"] = parse_r(len(self.blocks), self.r)
        x = self.forward_features(x, batch_size=batch_size, pos_embed=pos_embed, num_tokens=num_tokens)
        x = self.fc_dropout(x)
        x = self.head(x).unsqueeze(0)
        return x