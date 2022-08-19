# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class ChannelSpatialAttention(nn.Module):
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
        loss_weight=4e-3,
        policy='feature-based'
    ):
        super(ChannelSpatialAttention, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.policy = policy

    def dist2(self, tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
        diff = (tensor_a - tensor_b) ** 2
        if attention_mask is not None:
            diff = diff * attention_mask
        if channel_attention_mask is not None:
            diff = diff * channel_attention_mask
        diff = torch.sum(diff) ** 0.5
        return diff

    def spatial_mask(self, feature):
        attention_mask = torch.mean(torch.abs(feature), [1], keepdim=True)
        size = attention_mask.size()
        attention_mask = attention_mask.view(feature.size(0), -1)
        attention_mask = torch.softmax(attention_mask / self.tau, dim=1) * size[-1] * size[-2]
        attention_mask = attention_mask.view(size)
        return attention_mask

    def channel_mask(self, feature):
        attention_mask = torch.mean(torch.abs(feature), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
        size = attention_mask.size()
        attention_mask = attention_mask.view(feature.size(0), -1)  # 2 x 256
        attention_mask = torch.softmax(attention_mask / self.tau, dim=1) * 256
        attention_mask = attention_mask.view(size)  # 2 x 256 -> 2 x 256 x 1 x 1
        return attention_mask 


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
        s_ratio = 1.0
        c_ratio = 1.0
        
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        assert grads_S.shape[-2:] == grads_T.shape[-2:]
        assert preds_S.shape[-2:] == grads_T.shape[-2:]
        N, C, H, W = preds_S.shape

        if self.policy == 'feature-based':
            sum_attention_mask = (self.spatial_mask(preds_T) + self.spatial_mask(preds_S) * s_ratio) / (1 + s_ratio)
            sum_attention_mask = sum_attention_mask.detach()
            c_sum_attention_mask = (self.channel_mask(preds_T) + self.channel_mask(preds_S) * c_ratio) / (1 + c_ratio)
            c_sum_attention_mask = c_sum_attention_mask.detach()
        elif self.policy == 'gradient-based':
            sum_attention_mask = (self.spatial_mask(grads_T) + self.spatial_mask(grads_S) * s_ratio) / (1 + s_ratio)
            sum_attention_mask = sum_attention_mask.detach()
            c_sum_attention_mask = (self.channel_mask(grads_T) + self.channel_mask(grads_S) * c_ratio) / (1 + c_ratio)
            c_sum_attention_mask = c_sum_attention_mask.detach()
        elif self.policy == 't-gradient-based':
            sum_attention_mask = self.spatial_mask(grads_T)
            sum_attention_mask = sum_attention_mask.detach()
            c_sum_attention_mask = self.channel_mask(grads_T)
            c_sum_attention_mask = c_sum_attention_mask.detach()
        elif self.policy == 's-gradient-based':
            sum_attention_mask = self.spatial_mask(grads_S)
            sum_attention_mask = sum_attention_mask.detach()
            c_sum_attention_mask = self.channel_mask(grads_S)
            c_sum_attention_mask = c_sum_attention_mask.detach()
        elif self.policy == 'mixed':
            sum_attention_mask = (self.spatial_mask(grads_T) + self.spatial_mask(preds_S) * s_ratio) / (1 + s_ratio)
            sum_attention_mask = sum_attention_mask.detach()
            c_sum_attention_mask = (self.channel_mask(grads_T) + self.channel_mask(preds_S) * c_ratio) / (1 + c_ratio)
            c_sum_attention_mask = c_sum_attention_mask.detach()
        elif self.policy == 'added':
            sum_attention_mask = (self.spatial_mask(grads_T) + self.spatial_mask(preds_T) + self.spatial_mask(preds_S) * s_ratio) / (2 + s_ratio)
            sum_attention_mask = sum_attention_mask.detach()
            c_sum_attention_mask = (self.channel_mask(grads_T) + self.channel_mask(preds_T) + self.channel_mask(preds_S) * c_ratio) / (2 + c_ratio)
            c_sum_attention_mask = c_sum_attention_mask.detach()
        elif self.policy == 'plain':
            sum_attention_mask = None
            c_sum_attention_mask = None
        elif self.policy == 'grad-matching':
            kd_grad_loss = self.dist2(
                grads_T, 
                grads_S,
                attention_mask=None, 
                channel_attention_mask=None) * 1e4
            kd_feat_loss = self.dist2(
                preds_T, 
                preds_S,
                attention_mask=None, 
                channel_attention_mask=None) * 0.01
            loss = kd_feat_loss + kd_grad_loss
            loss = self.loss_weight * loss / N
            return loss
        elif self.policy == 'grad-matching-only':
            kd_grad_loss = self.dist2(
                grads_T, 
                grads_S,
                attention_mask=None, 
                channel_attention_mask=None) * 1e4
            loss = kd_grad_loss
            loss = self.loss_weight * loss / N
            return loss
        else:
            print('policy [%d] not support' % self.policy)

        kd_feat_loss = self.dist2(
            preds_T, 
            preds_S,
            attention_mask=sum_attention_mask, 
            channel_attention_mask=c_sum_attention_mask) * 0.01

        loss = kd_feat_loss


        # kd_channel_loss = torch.dist(
        #     torch.mean(preds_T, [2, 3]), 
        #     torch.mean(preds_S, [2, 3]),
        #     )

        # kd_spatial_loss = torch.dist(
        #     torch.mean(preds_T, [1]).view(N,1,H,W), 
        #     torch.mean(preds_S, [1]).view(N,1,H,W),
        #     )

        # loss =                kd_channel_loss + kd_spatial_loss
        # loss = kd_feat_loss +                   kd_spatial_loss
        # loss = kd_feat_loss + kd_channel_loss 
        # loss = kd_feat_loss + kd_channel_loss + kd_spatial_loss

        loss = self.loss_weight * loss / N ########################### should normalize each one??

        return loss
