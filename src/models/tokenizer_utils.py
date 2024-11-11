from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class MixupConfig:
    mixup_alpha: float
    cutmix_alpha: float
    prob: float
    switch_prob: float
    mode: str
    label_smoothing: float
    num_classes: int
    cutmix_minmax: Optional[Tuple[float, float]] = None

@dataclass
class RandomErasingConfig:
    probability: float
    mode: str
    min_count: int
    device: str

@dataclass
class RandAugmentConfig:
    magnitude: int
    num_ops: int

@dataclass
class TokenizerConfig:
    drop_policy: str
    drop_param: float
    encode_length: bool
    mixup_config: Optional[MixupConfig] = None
    re_config: Optional[RandomErasingConfig] = None
    ra_config: Optional[RandAugmentConfig] = None