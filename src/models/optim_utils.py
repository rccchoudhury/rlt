"""
Several utils and configs for setting up
learning rate, schedulers, weight decay and 
layer decay.
"""

from dataclasses import dataclass
import ipdb
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

# @dataclass
# class OptimizerConfig:
#     lr: float
#     # WD
#     weight_decay: float
#     # DO we use a weight decay schedule?
#     # Alpha to use for layer decay
#     layer_decay: float

@dataclass
class CosineSchedulerConfig:
    warmup_epochs: int
    total_epochs: int
    # The value to warm up to.
    lr: float   
    # The value to cosine decay to.
    end_lr: float
    # the value to start at.
    start_lr: float

def cosine_scheduler(optimizer: Optimizer,
                      config: CosineSchedulerConfig) -> LRScheduler:
    """Setup the learning rate scheduler for training."""
    assert hasattr(config, "total_steps")
    assert hasattr(config, "batch_size")

    warmup_epochs = 5
    total_epochs =config.total_epochs
    total_steps = config.total_steps
    steps_per_epoch = total_steps // total_epochs
    warmup_steps = warmup_epochs * steps_per_epoch
    # Decrease LR by factor (following AVION)
    # TODO: factor should be sqrt?? [cite]
    factor = config.batch_size / 256
    target_lr = config.lr * factor
    base_lr = config.end_lr * factor
    initial_lr = config.start_lr * factor

    def cosine_with_warmup(step):
        if step < warmup_steps:
            # Linear warmup from initial_lr to target_lr
            return initial_lr + (step / warmup_steps) * (target_lr - initial_lr)
        else:
            # Cosine annealing from target_lr to base_lr
            adjusted_step = step - warmup_steps
            total_adjusted_steps = total_steps - warmup_steps
            return base_lr + (0.5 * (target_lr - base_lr) * (1 + math.cos(math.pi * adjusted_step / total_adjusted_steps)))


    # Define the warm-up scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lambda x: cosine_with_warmup(x))

    return scheduler


class LayerDecayValueAssigner:
    def __init__(self, values):
        self.values = values
        self.num_layers = len(self.values)

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        if var_name in ['cls_token', 'mask_token', 'pos_embed']:
            return 0
        elif var_name.startswith('patch_embed'):
            return 0
        elif var_name.startswith('rel_pos_bias') or var_name == 'length_embed':
            return self.num_layers - 1
        elif var_name.startswith('blocks'):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return self.num_layers - 1
        
def setup_layer_decay(model, ld_assigner, weight_decay=1e-5, skip_list=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        layer_id = ld_assigner.get_layer_id(name)
        group_name = "layer_%d_%s" % (layer_id, group_name)


        if group_name not in parameter_group_names:
            scale = ld_assigner.get_scale(layer_id)

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    #print("Param groups:", parameter_group_names)
    return list(parameter_group_vars.values())