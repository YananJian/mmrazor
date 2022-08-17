# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

from ..builder import DISTILLERS, MODELS, build_loss
from .base import BaseDistiller
import pdb

from torch.autograd import Function

class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * -ctx.alpha
        return output, None

def grad_reverse(x,alpha):
    return GradReverse.apply(x,alpha)


class Discriminator(nn.Module):
    def __init__(self, outputs_size, K = 2):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size//K, kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv2 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size//K, kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv3 = nn.Conv2d(in_channels=outputs_size, out_channels=2, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        #x = x[:,:,None,None]
        out = F.leaky_relu(self.conv1(x), 0.02)
        out = F.leaky_relu(self.conv2(out), 0.02)
        out = F.leaky_relu(self.conv3(out), 0.02)
        #pdb.set_trace()
        #out = out.view(out.size(0), -1)
        return out


class Discriminators(nn.Module):
    def __init__(self, output_dims, grl):
        super(Discriminators, self).__init__()
        self.discriminators = [Discriminator(i) for i in output_dims]
        self.grl = grl
    
    def forward(self, x):
        if self.grl == True:
            out = [self.discriminators[i](grad_reverse(x[i], 1.0)) for i in range(len(self.discriminators))]
        else:
            out = [self.discriminators[i](x[i]) for i in range(len(self.discriminators))]

        return out


class DiscriminatorModules(nn.Module):
    def __init__(self, models):
        super(DiscriminatorModules, self).__init__()
        self.models = models

    def forward(self, outputs, targets):
        inputs = [torch.cat((i,j),0) for i, j in zip(outputs, targets)]
        outputs = self.models(inputs)
        return outputs


@DISTILLERS.register_module()
class SingleTeacherDistiller(BaseDistiller):
    """Distiller with single teacher.

    Args:
        teacher (dict): The config dict for teacher.
        teacher_trainable (bool): Whether the teacher is trainable.
            Default: False.
        teacher_norm_eval (bool): Whether to set teacher's norm layers to eval
            mode, namely, freeze running stats (mean and var). Note: Effect on
            Batch Norm and its variants only. Default: True.
        components (dict): The details of the distillation. It usually includes
            the module names of the teacher and the student, and the losses
            used in the distillation.
    """

    def __init__(self,
                 teacher,
                 teacher_trainable=False,
                 teacher_norm_eval=True,
                 components=tuple(),
                 **kwargs):
        super().__init__(**kwargs)
        self.teacher_trainable = teacher_trainable
        self.teacher_norm_eval = teacher_norm_eval
        self.teacher = self.build_teacher(teacher)

        self.components = components
        self.losses = nn.ModuleDict()
        self.align_modules = nn.ModuleDict()
        self.discriminator_modules = nn.ModuleDict()

        # Record the featuremaps that need to calculate the distillation loss.
        self.student_outputs = dict()
        self.teacher_outputs = dict()

        for i, component in enumerate(self.components):
            student_module_name = component['student_module']
            teacher_module_name = component['teacher_module']
            # The type of every student_output is a list by default, because
            # some modules will execute multiple forward calculations, such as
            # the shareable head in Retinanet
            self.student_outputs[student_module_name] = list()
            self.teacher_outputs[teacher_module_name] = list()
            # If the number of featuremap channels of student and teacher are
            # inconsistent, they need to be aligned by a 1x1 convolution
            align_module_cfg = getattr(component, 'align_module', None)
            if align_module_cfg is not None:
                align_module_name = f'component_{i}'
                align_module = self.build_align_module(align_module_cfg)
                self.align_modules[align_module_name] = align_module
            
            discriminator_module_cfg = getattr(component, 'discriminator_module', None)
            if discriminator_module_cfg is not None:
                discriminator_modules = self.build_discriminator_modules(discriminator_module_cfg)
                discriminator_modules = discriminator_modules.to("cuda")
                for j, d in enumerate(discriminator_modules.models.discriminators):
                    #self.register_parameter(f'discriminator_{j}', d.parameters())
                    d = d.to("cuda")
                discriminator_module_name = f'component_{i}'
                self.discriminator_modules[discriminator_module_name] = discriminator_modules
                    
            # Multiple losses can be calculated at the same location
            for loss in component.losses:
                loss_cfg = loss.copy()
                loss_name = loss_cfg.pop('name')
                self.losses[loss_name] = build_loss(loss_cfg)

    def build_teacher(self, cfg):
        """Build a model from the `cfg`."""

        teacher = MODELS.build(cfg)

        return teacher

    def build_align_module(self, cfg):
        """Build ``align_module`` from the `cfg`.

        ``align_module`` is needed when the number of channels output by the
        teacher module is not equal to that of the student module, or for some
        other reasons.

        Args:
            cfg (dict): The config dict for ``align_module``.
        """

        in_channels = cfg.student_channels
        out_channels = cfg.teacher_channels
        if cfg.type == 'conv2d':
            align_module = nn.Conv2d(in_channels, out_channels, 1)
        elif cfg.type == 'linear':
            align_module = nn.Linear(in_channels, out_channels)
        return align_module

    def build_discriminator_modules(self, cfg):
        """Build ``discriminator_modules`` from the `cfg`.

        Args:
            cfg (dict): The config dict for ``discriminator_module``.
        """
        discriminators = Discriminators(output_dims=cfg.in_channels, grl=True)
        d_modules = DiscriminatorModules(discriminators)
        return d_modules

    def prepare_from_student(self, student):
        """Registers a global forward hook for each teacher module and student
        module to be used in the distillation.

        Args:
            student (:obj:`torch.nn.Module`): The student model to be used
                in the distillation.
        """

        # Record the mapping relationship between student's modules and module
        # names.
        self.student_module2name = {}
        for name, module in student.model.named_modules():
            self.student_module2name[module] = name
        self.student_name2module = dict(student.model.named_modules())

        # Record the mapping relationship between teacher's modules and module
        # names.
        self.teacher_module2name = {}
        for name, module in self.teacher.named_modules():
            self.teacher_module2name[module] = name
        self.teacher_name2module = dict(self.teacher.named_modules())


        # Register forward hooks for modules that need to participate in loss
        # calculation.
        for component in self.components:
            student_module_name = component['student_module']
            teacher_module_name = component['teacher_module']
            discriminator_module_name = component['discriminator_module']

            student_module = self.student_name2module[student_module_name]
            teacher_module = self.teacher_name2module[teacher_module_name]

            student_module.register_forward_hook(
                self.student_forward_output_hook)
            teacher_module.register_forward_hook(
                self.teacher_forward_output_hook)

    def teacher_forward_output_hook(self, module, inputs, outputs):
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        if self.training:
            self.teacher_outputs[self.teacher_module2name[module]].append(
                outputs)

    def student_forward_output_hook(self, module, inputs, outputs):
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        if self.training:
            self.student_outputs[self.student_module2name[module]].append(
                outputs)

    def reset_outputs(self, outputs):
        """Reset the teacher's outputs or student's outputs."""
        for key in outputs.keys():
            outputs[key] = list()

    def exec_teacher_forward(self, data):
        """Execute the teacher's forward function.

        After this function, the teacher's featuremaps will be saved in
        ``teacher_outputs``.
        """

        # Convert the context manager's mode to teacher.
        self.reset_ctx_teacher_mode(True)
        # Clear the saved data of the last forward。
        self.reset_outputs(self.teacher_outputs)

        if self.teacher_trainable:
            output = self.teacher(**data)
        else:
            with torch.no_grad():
                output = self.teacher(**data)
        return output

    def exec_student_forward(self, student, data):
        """Execute the teacher's forward function.

        After this function, the student's featuremaps will be saved in
        ``student_outputs``.
        """
        # Convert the context manager's mode to teacher.
        self.reset_ctx_teacher_mode(False)
        # Clear the saved data of the last forward。
        self.reset_outputs(self.student_outputs)

        output = student(**data)
        return output

    def train(self, mode=True):
        """Set distiller's forward mode."""
        super(SingleTeacherDistiller, self).train(mode)
        if mode and self.teacher_norm_eval:
            for m in self.teacher.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def get_teacher_outputs(self, teacher_module_name):
        """Get the outputs according module name."""
        return self.teacher_outputs[teacher_module_name]

    def compute_distill_loss(self, data=None):
        """Compute the distillation loss."""

        losses = dict()

        for i, component in enumerate(self.components):
            # Get the student's outputs.
            student_module_name = component['student_module']
            student_outputs = self.student_outputs[student_module_name]

            # Align student output's channels with teacher.
            align_module_name = f'component_{i}'
            if align_module_name in self.align_modules:
                align_module = self.align_modules[align_module_name]
                student_outputs = [
                    align_module(s_out) for s_out in student_outputs
                ]

            # Get the teacher's outputs.
            teacher_module_name = component['teacher_module']
            teacher_outputs = self.get_teacher_outputs(teacher_module_name)

            discriminator_module_name = f'component_{i}'
            if discriminator_module_name in self.discriminator_modules:
                discriminator_module = self.discriminator_modules[discriminator_module_name]
                discriminator_outputs = discriminator_module(student_outputs, teacher_outputs)
            
            # One module maybe have N outputs, such as the shareable head in
            # RetinaNet.
            for out_idx, (s_out, t_out, d_out) in enumerate(
                    zip(student_outputs, teacher_outputs, discriminator_outputs)):

                for loss in component.losses:
                    loss_module = self.losses[loss.name]
                    loss_name = f'{loss.name}.{out_idx}'
                    # TODO ugly implementation.
                    # Pass the gt_label to loss function.
                    # Only used by WSLD.
                    loss_module.current_data = data
                    if loss.name == 'loss_discriminator':
                        losses[loss_name] = loss_module(s_out, t_out, d_out)
                    else:
                        losses[loss_name] = loss_module(s_out, t_out)
                    loss_module.current_data = None
        return losses
