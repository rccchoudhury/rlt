import torch
from typing import Tuple

from einops import rearrange
from torch import Tensor
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms import v2, InterpolationMode
#from pytorchvideo.transforms import RandAugment

import ipdb
from .mixup import Mixup
from .static_token_utils import batched_find_idxs_to_keep, random_droptoken, batched_get_token_lengths
from .random_erasing import RandomErasing

def get_sinusoid_encoding(n_position: int, embed_dims: int, base: int = 10000) -> Tensor:
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
    vec = torch.pow(base, -vec).view(1, -1)

    sinusoid_table = torch.arange(n_position).view(-1, 1) * vec
    sinusoid_table[:, 0::2].sin_()  # dim 2i
    sinusoid_table[:, 1::2].cos_()  # dim 2i+1

    sinusoid_table = sinusoid_table.to(torch.float32)

    return sinusoid_table.unsqueeze(0)


class Tokenizer(nn.Module):
    def __init__(self, 
                 drop_policy: str='rlt', 
                 drop_param: float=0.1,
                 encode_length: bool=False,
                 num_frames: int=16,
                 tubelet_size: int=2,
                 patch_size: Tuple[int, int]=(16, 16),
                 frame_size: Tuple[int, int]=(224, 224),
                 transform: nn.Module=None,
                 embed_dims: int=768,
                 mixup_fn: Mixup=None,
                 random_erase_fn: RandomErasing=None,
                 rand_aug_fn=None) -> None:
        super().__init__()
        self.do_drop = drop_policy == 'rlt'
        self.drop_param = drop_param
        self.encode_length = encode_length
        self.random_drop = drop_policy == 'random'
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.total_tokens= (num_frames // tubelet_size) * (frame_size[0] // patch_size[0]) * (frame_size[1] // patch_size[1])
        # assert self.total_tokens == 1568 # not compatible with no spatial patch
        self.embed_dims = embed_dims
        self.pos_embed = get_sinusoid_encoding(n_position=self.total_tokens, embed_dims=embed_dims)[0]
        self.length_embed = get_sinusoid_encoding(n_position=num_frames//tubelet_size, embed_dims=embed_dims, base=1000)[0]
        self.desired_shape = (num_frames // tubelet_size, frame_size[0] // patch_size[0], frame_size[1] // patch_size[1])
        self.val_transform =  nn.Sequential(
            v2.Resize(224, interpolation=InterpolationMode.BILINEAR),
            v2.CenterCrop(224),
        )
        self.rand_aug_fn = rand_aug_fn
        self.transform=transform
        self.mixup_fn = mixup_fn
        self.random_erase_fn = random_erase_fn

    def collate_tokens(self, crops: torch.Tensor, 
                    token_mask: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects only the applicable tokens, based on dropping criteria.
        If no dropping, simply concatenates the sequences to produce
        a single batch.
        """
        assert len(token_mask.shape) == 2
        assert len(crops.shape) == 6
        assert crops.shape[0] == token_mask.shape[0]
        # Find the biggest number of tokens.
        num_valid_tokens = torch.sum(token_mask, dim=1)
        padded_tokens = []
        pos_embeds = []
        # CURRENT HACK: GET THE POS EMBED SHAPE TO BE RIGHT...
        for i in range(crops.shape[0]):
            # Select the valid tokens and appropriate positional embeddings.
            valid_tokens = crops[i][token_mask[i]]
            valid_pos = self.pos_embed[token_mask[i]]

            padded_tokens.append(valid_tokens)
            pos_embeds.append(valid_pos)
        
        return torch.cat(padded_tokens), num_valid_tokens, torch.cat(pos_embeds)

    def forward(self, frames: Tensor, targets: Tensor, is_training: bool=False) -> Tensor:
        B, T, H, W, C = frames.shape
        assert frames.device != 'cpu'
        if self.pos_embed.device != frames.device:
            self.pos_embed = self.pos_embed.to(frames.device)
            self.length_embed = self.length_embed.to(frames.device)
        #ipdb.set_trace()
        frames = frames.reshape(B, -1, self.num_frames, H, W, C).flatten(0, 1)
        
        assert len(frames.shape) == 5
        # it's batch_size * num_crops.
        B = frames.shape[0]
        frames = frames.permute(0, 1, 4, 2, 3)

        # This runs random augmentation, but applies the same aug to the 
        # whole batch, which isn't ideal.
        # if is_training and self.rand_aug_fn is not None:
        #     ipdb.set_trace()
        #     frames = frames.flatten(0, 1)
        #     frames = self.rand_aug_fn(frames)
        #     frames = frames.reshape(B, -1, *frames.shape[1:])

        # Run imagenet normalization + convert to float.
        if self.transform is not None:
            frames = self.transform(frames)
        # Run augmentation.
        if is_training:
            # Switch to B, C, T, H, W
            if self.random_erase_fn is not None:
                frames = frames.transpose(1, 2)
                frames = self.random_erase_fn(frames)
                # Switch back to B, T, C, H, W
                frames = frames.transpose(1, 2)
            if self.mixup_fn is not None and frames.shape[0] % 2 == 0:
                frames, targets = self.mixup_fn(frames, targets)


        split_crops = frames.transpose(1, 2)
        # Obtain token masks if we do token dropping.
        token_lengths = None
        if self.do_drop:
            # Find static tokens.
            token_mask = batched_find_idxs_to_keep(split_crops, 
                                                   threshold=self.drop_param, 
                                                   tubelet_size=self.tubelet_size,
                                                   patch_size=self.patch_size[0])
            #ipdb.set_trace()
            if self.encode_length:
                token_lengths = batched_get_token_lengths(token_mask, batch_size=B, input_shape=self.desired_shape)
        else:
            token_mask = torch.ones((B, self.total_tokens), dtype=torch.bool)
            
        
        split_crops = rearrange(
            split_crops, 
            "b c (t p0) (h p1) (w p2) -> b (t h w) c p0 p1 p2",
            p0=self.tubelet_size,
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )
        #ipdb.set_trace()
        if self.random_drop:# and is_training:
            # Droptoken for evaluation baseline.
            token_mask = random_droptoken(split_crops, 
                                          n_tokens=self.total_tokens, 
                                          p=self.drop_param)
        # ONLY INCLUDE THIS STEP HERE FOR THE MULTI-CLIP EVALUATION.
        split_crops, num_tokens, pos_embeds = self.collate_tokens(split_crops, token_mask)

        # Return output dict.
        output_dict = {}
        output_dict['split_crops'] = split_crops
        output_dict['num_tokens'] = num_tokens
        output_dict['pos_embeds'] = pos_embeds
        output_dict['targets'] = targets
        output_dict['token_mask'] = token_mask
        output_dict['frames'] = frames
        if self.encode_length:
            output_dict['token_lengths'] = token_lengths

        return output_dict