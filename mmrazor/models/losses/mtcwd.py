# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class MTChannelWiseDivergence(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
        channel_in=256,
        policy="learn_mask"
    ):
        super(MTChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.policy = policy
        print("Currently using multi-teacher policy:", self.policy)


    def selective_aggregate(self, preds_S, preds_T, align_layers=None):

        N, C, H, W = preds_S.shape

        if len(preds_T) == 1:
            return preds_T[0]

        if self.policy == 'avg':
            # shape after stacking: (2, N, C, H, W)
            # shape after mean: (N, C, H, W)
            preds_T = torch.mean(torch.stack(preds_T), dim=0)

        elif self.policy == 'dropout':
            dice = torch.randn(len(preds_T))
            idx = torch.argmax(dice).cuda()
            preds_T = preds_T[idx]

        elif self.policy == 'learn_mask':
            new_preds_T = []
            for T in preds_T:
                diff = torch.abs(T - preds_S)
                mask = align_layers(diff).view(N, C, 1, 1)
                new_preds_T.append(mask * T)

            preds_T = torch.mean(torch.stack(new_preds_T), dim=0)

        return preds_T

    def forward(self, preds_S, preds_T, align_layers=None):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).
            align_layers: used for masking multi-teacher channels, 
                only required when policy == "learn_mask".

        Return:
            torch.Tensor: The calculated loss value.
        """
        
        # print("--------------preds_S --- len() --- [0].shape---------------", preds_S.shape)
        # print("--------------preds_T --- len() --- [0].shape---------------", len(preds_T), preds_T[0].shape)

        preds_T = self.selective_aggregate(preds_S, preds_T, align_layers)

        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape
        
        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        info_T = softmax_pred_T * logsoftmax(preds_T.view(-1, W * H) / self.tau)
        
        loss = torch.sum(info_T -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)

        loss = self.loss_weight * loss / (C * N)

        return loss
