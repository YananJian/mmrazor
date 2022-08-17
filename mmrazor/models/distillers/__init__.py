# Copyright (c) OpenMMLab. All rights reserved.
from .self_distiller import SelfDistiller
from .single_teacher import SingleTeacherDistiller
from .multi_teacher import MultiTeacherDistiller

__all__ = ['SelfDistiller', 'SingleTeacherDistiller',
	'MultiTeacherDistiller'
]
