from typing import Tuple
import torch
import torch.nn.functional as F

import ipdb

def batched_find_idxs_to_keep(x: torch.Tensor, 
                              threshold: int=2, 
                              tubelet_size: int=2,
                              patch_size: int=16) -> torch.Tensor:
    """
    Find the static tokens in a video tensor, and return a mask
    that selects tokens that are not repeated.

    Args:
     - x (torch.Tensor): A tensor of shape [B, C, T, H, W].
     - threshold (int): The mean intensity threshold for considering
            a token as static.
     - tubelet_size (int): The temporal length of a token.
    Returns:
     - mask (torch.Tensor): A bool tensor of shape [B, T, H, W] 
        that selects tokens that are not repeated.

    """
    # Ensure input has the format [B, C, T, H, W]
    assert len(x.shape) == 5, "Input must be a 5D tensor"
    #ipdb.set_trace()
    # Convert to float32 if not already
    x = x.type(torch.float32)
    
    # Calculate differences between frames with a step of tubelet_size, ensuring batch dimension is preserved
    # Compare "front" of first token to "back" of second token
    diffs = x[:, :, (2*tubelet_size-1)::tubelet_size] - x[:, :, :-tubelet_size:tubelet_size]
    # Ensure nwe track negative movement.
    diffs = torch.abs(diffs)
    
    # Apply average pooling over spatial dimensions while keeping the batch dimension intact
    avg_pool_blocks = F.avg_pool3d(diffs, (1, patch_size, patch_size))
    # Compute the mean along the channel dimension, preserving the batch dimension
    avg_pool_blocks = torch.mean(avg_pool_blocks, dim=1, keepdim=True)
    # Create a dummy first frame for each item in the batch
    first_frame = torch.ones_like(avg_pool_blocks[:, :, 0:1]) * 255
    # Concatenate the dummy first frame with the rest of the frames, preserving the batch dimension
    avg_pool_blocks = torch.cat([first_frame, avg_pool_blocks], dim=2)
    # Determine indices to keep based on the threshold, ensuring the operation is applied across the batch
    
    keep_idxs = avg_pool_blocks.squeeze(1) > threshold  
    # Flatten out everything but the batch dimension
    keep_idxs = keep_idxs.flatten(1)
    #ipdb.set_trace()
    return keep_idxs


def batched_get_token_lengths(token_mask: torch.Tensor, batch_size:int, input_shape: Tuple) -> torch.Tensor:
    """
    Takes in a binary tensor for which tokens are retained and returns the 
    duration (length along T axis) for each token in the batch.

    Takes input 
        token_mask: shape (B, T, H, W)

    Returns 
        lengths: shape (B, n_toks)
    """
    #ipdb.set_trace()
    input_tensor = token_mask.reshape((batch_size,) + input_shape)
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    B, H, W, T = input_tensor.shape
    # Concat with ones at the end, so terminal tokens have length 1.
    concat_tensor = torch.cat([input_tensor, 
                             torch.ones((B, H, W, 1), 
                                        dtype=torch.uint8, device=input_tensor.device)
                                        ], dim=-1)\
    #ipdb.set_trace()
    range_tensor = torch.arange(T+1, device=input_tensor.device).reshape((1, 1, 1, T+1))
    range_tensor = range_tensor.repeat(B, H, W, 1)
    diffs = range_tensor[concat_tensor.bool()].diff()
    diffs = diffs[diffs > 0]
    #ipdb.set_trace()
    return diffs

def random_droptoken(x: torch.Tensor, n_tokens: int, p: float=0.5) -> torch.Tensor:
    """
    For each clip in the batch, drop a random p of tokens.
    """
    assert 0 <= p <= 1, "p must be between 0 and 1"
    #ipdb.set_trace()
    B, n_tokens = x.shape[0], x.shape[1]
    
    random_values = torch.rand((B, n_tokens), dtype=torch.float)

    # Determine the threshold for each row that corresponds to the top p fraction of values
    threshold = torch.kthvalue(random_values, k=int(n_tokens * p), dim=1, keepdim=True)[0]

    # Create a mask where values below the threshold are selected
    mask = random_values < threshold

    return mask