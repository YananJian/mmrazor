# Copyright (c) OpenMMLab. All rights reserved.
from .cwd import ChannelWiseDivergence
from .kl_divergence import KLDivergence
from .relational_kd import AngleWiseRKD, DistanceWiseRKD
from .weighted_soft_label_distillation import WSLD

from .mtcwd import MTChannelWiseDivergence

__all__ = [
    'ChannelWiseDivergence', 'KLDivergence', 'AngleWiseRKD', 'DistanceWiseRKD',
    'WSLD', 'MTChannelWiseDivergence',
]
