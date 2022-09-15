# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import build_optimizer
import torch


def build_optimizers(model, cfgs):
    """Build multiple optimizers from configs. If `cfgs` contains several dicts
    for optimizers, then a dict for each constructed optimizers will be
    returned. If `cfgs` only contains one optimizer config, the constructed
    optimizer itself will be returned. For example, 1) Multiple optimizer
    configs:
    code-block::

        optimizer_cfg = dict(
            model1=dict(type='SGD', lr=lr),
            model2=dict(type='SGD', lr=lr))

    The return dict is
    ``dict('model1': torch.optim.Optimizer, 'model2': torch.optim.Optimizer)``
    2) Single optimizer config:
    .. code-block::

        optimizer_cfg = dict(type='SGD', lr=lr)

    The return is ``torch.optim.Optimizer``.
    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        cfgs (dict): The config dict of the optimizer.
    Returns:
        dict[:obj:`torch.optim.Optimizer`] | :obj:`torch.optim.Optimizer`:
            The initialized optimizers.
    """
    optimizers = {}
    if hasattr(model, 'module'):
        model = model.module
    # determine whether 'cfgs' has several dicts for optimizers
    is_dict_of_dict = True
    for key, cfg in cfgs.items():
        if not isinstance(cfg, dict):
            is_dict_of_dict = False

    if is_dict_of_dict:
        for key, cfg in cfgs.items():
            cfg_ = cfg.copy()
            module = getattr(model, key)
            optimizers[key] = build_optimizer(module, cfg_)
        return optimizers

    return build_optimizer(model, cfgs)
    '''
    optimizer_cfg = cfgs
    base_lr = optimizer_cfg['lr']
    base_wd = optimizer_cfg.get('weight_decay', None)
    params = [{'params': model.module.architecture.parameters(), 'lr': base_lr, 'weight_decay': base_wd}, 
          {'params': model.module.distiller.ta.parameters(), 'lr': base_lr*0.01, 'weight_decay': base_wd}]
    optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
    return optimizer_cls(params, **optimizer_cfg)
    '''
