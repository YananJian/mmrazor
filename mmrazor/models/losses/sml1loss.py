# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """PyTorch version of `ICLR2021.

    <https://openreview.net/pdf?id=uKhGRvM8QNH>`_.

    Code reference from: https://github.com/ArchipLab-LinfengZhang/Object-Detection-Knowledge-Distillation-ICLR2021

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
        policy='feature-based'
    ):
        super(SmoothL1Loss, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

        self.policy = policy
        self.loss = torch.nn.SmoothL1Loss(beta=2.0)

        self.feat_w = 1e1
        self.grad_w = 1e6

    def forward(self, preds_S, preds_T, grads_S, grads_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """

        # student's spatial and channel mix ratio
  
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        assert grads_S.shape[-2:] == grads_T.shape[-2:]
        assert preds_S.shape[-2:] == grads_T.shape[-2:]
        N, C, H, W = preds_S.shape

        layer_norm = nn.LayerNorm([C, H, W], elementwise_affine=False)

        if self.policy == 'feature-based':
            loss = self.feat_w * self.loss(preds_S, layer_norm(preds_T))
        elif self.policy == 'grad-matching-only':
            loss = self.grad_w * self.loss(layer_norm(grads_S), layer_norm(grads_T))
        elif self.policy == 'combined':
            loss = 0.5 * (
                self.feat_w * self.loss(preds_S, layer_norm(preds_T)) + 
                self.grad_w * self.loss(layer_norm(grads_S), layer_norm(grads_T))
                )
        elif self.policy == 'multiplied':
            loss = self.grad_w * self.loss(layer_norm(preds_S * grads_S), layer_norm(preds_T * grads_T))
        else:
            print('policy [%d] not support' % self.policy)


        loss = self.loss_weight * loss #### / N #############################should not divide by N anymore.

        return loss
