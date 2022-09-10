# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import DISTILLERS, MODELS, build_loss
from .base import BaseDistiller
import copy


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
                 assist=False,
                 assist_loss_mul=None,
                 assist_module=dict, 
                 **kwargs):
        super().__init__(**kwargs)
        self.teacher_trainable = teacher_trainable
        self.teacher_norm_eval = teacher_norm_eval
        self.teacher = self.build_teacher(teacher)

        self.components = components
        self.assist_module = assist_module
        self.losses = nn.ModuleDict()
        self.align_modules = nn.ModuleDict()

        # Record the featuremaps that need to calculate the distillation loss.
        self.student_outputs = dict()
        self.teacher_outputs = dict()
        self.assist_outputs = dict()

        self.student_feats = None

        self.student_grads = dict()
        self.teacher_grads = dict()

        self.teacher_roi_logits = list()

        self.temp_count = 0
        self.ta = None

        for i, component in enumerate(self.components):
            if component.get('student_module') is not None:
                student_module_name = component['student_module']
                self.student_outputs[student_module_name] = list()

            if component.get('assist_module') is not None:
                assist_module_name = component['assist_module']
                self.assist_outputs[assist_module_name] = list()

            teacher_module_name = component['teacher_module']
            self.teacher_outputs[teacher_module_name] = list()

            #self.student_grads[student_module_name] = list()
            #self.teacher_grads[teacher_module_name] = list()

            # If the number of featuremap channels of student and teacher are
            # inconsistent, they need to be aligned by a 1x1 convolution
            align_module_cfg = getattr(component, 'align_module', None)
            if align_module_cfg is not None:
                align_module_name = f'component_{i}'
                align_module = self.build_align_module(align_module_cfg)
                self.align_modules[align_module_name] = align_module

            # Multiple losses can be calculated at the same location
            for loss in component.losses:
                loss_cfg = loss.copy()
                loss_name = loss_cfg.pop('name')
                self.losses[loss_name] = build_loss(loss_cfg)
        
        if assist: 
            self.ta = self.build_assist()
            ta_modules = [x for x in self.ta.modules()]
            for x in ta_modules:
                x.requires_grad_() 
            self.ta.init_weights()
        self.assist = assist
        self.assist_loss_mul = assist_loss_mul
        self.adjust_device = False


    def build_teacher(self, cfg):
        """Build a model from the `cfg`."""

        teacher = MODELS.build(cfg)

        return teacher

    def build_assist(self):
        #assist_head = HEADS.build(assist)
        assist_head = copy.deepcopy(self.teacher.bbox_head)
        return assist_head

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
        self.assist_module2name = {}
        for name, module in self.ta.named_modules():
            self.assist_module2name[module] = name
        self.assist_name2module = dict(self.ta.named_modules())

        for component in self.components:
            if component.get('student_module') is not None: 
                student_module_name = component['student_module']
                student_module = self.student_name2module[student_module_name]
                student_module.register_forward_hook(self.student_forward_output_hook)

            if component.get('assist_module') is not None: 
                assist_module_name = component['assist_module']
                #assist_module = eval('%s.%s'%('self.ta', assist_module_name))
                assist_module = self.assist_name2module[assist_module_name]
                assist_module.register_forward_hook(self.assist_forward_output_hook)

            teacher_module_name = component['teacher_module']
            teacher_module = self.teacher_name2module[teacher_module_name]
            teacher_module.register_forward_hook(self.teacher_forward_output_hook)

        if self.assist:
            student_module_name = self.assist_module['student_module']
            teacher_module_name = self.assist_module['teacher_module']
            student_module = self.student_name2module[student_module_name]
            teacher_module = self.teacher_name2module[teacher_module_name]

            student_module.register_forward_hook(
                self.assist_student_forward_output_hook)
            
        #self.teacher.roi_head.bbox_head.register_forward_hook(self.teacher_roi_output_hook)
    def assist_forward_output_hook(self, module, inputs, outputs):
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        if self.training:
            self.assist_outputs[self.assist_module2name[module]].append(
                outputs)
            # student FPN is 5 layers and starts from ([2, 256, 200, 304])
            # in order to match teacher Bbox_head inputs, need to use all student FPN layers except for the first FPN.

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

    def teacher_roi_output_hook(self, module, inputs, outputs):
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        self.teacher_roi_logits.append(outputs[0])


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
            # student FPN is 5 layers and starts from ([2, 256, 200, 304])
            # in order to match teacher Bbox_head inputs, need to use all student FPN layers except for the first FPN.
            #import pdb;pdb.set_trace()

    def assist_student_forward_output_hook(self, module, inputs, outputs):
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        if self.training:
            # student FPN is 5 layers and starts from ([2, 256, 200, 304])
            # in order to match teacher Bbox_head inputs, need to use all student FPN layers except for the first FPN.
            #import pdb;pdb.set_trace()
            self.student_feats = inputs

    def teacher_grad_hook(self, outputs, loss_types=['loss_cls', 'loss_bbox']):
        outputs = [outputs[loss] for loss in loss_types]

        modules = [k for k in self.teacher_outputs.keys()]
        inputs = [self.teacher_outputs[m][0] for m in self.teacher_outputs]
        # print(inputs)
        grads = torch.autograd.grad(outputs, inputs, retain_graph=True, allow_unused=True)
        for i, grad in enumerate(grads):
            self.teacher_grads[modules[i]] = grads[i]
        
    def student_grad_hook(self, outputs, loss_types=['loss_cls', 'loss_bbox']):
        outputs = [outputs[loss] for loss in loss_types]

        modules = [k for k in self.student_outputs.keys()]
        inputs = [self.student_outputs[m][0] for m in modules]

        grads = torch.autograd.grad(outputs, inputs, retain_graph=True, allow_unused=True)
        for i, grad in enumerate(grads):
            self.student_grads[modules[i]] = grads[i]

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
        # Clear the saved data of the last forward.
        self.reset_outputs(self.teacher_outputs)
        #self.reset_outputs(self.teacher_grads)
        #self.teacher_roi_logits = list()

        if self.teacher_trainable:
            output = self.teacher(**data)
        else:
            with torch.no_grad():
                output = self.teacher(**data)
            #output = self.teacher(**data)
            #self.teacher_grad_hook(output)
            # print(data['img'].shape)
            # m = [k for k in self.teacher_outputs.keys()]
            # print(output, self.teacher_outputs[m[0]][0].shape)
            # grad = torch.autograd.grad(output['loss_cls'], self.teacher_outputs[m[0]][0])
            # print(grad[0].shape)
            # torch.save({"img": data['img'], "feat":self.teacher_outputs[m[0]][0], "grad": self.teacher_grads[m[0]]}, '/home/frank/Desktop/notebooks/tensors_%d.pt' % self.temp_count)
            #self.temp_count += 1
        return output

    def exec_student_forward(self, student, data):
        """Execute the teacher's forward function.

        After this function, the student's featuremaps will be saved in
        ``student_outputs``.
        """
        # Convert the context manager's mode to teacher.
        self.reset_ctx_teacher_mode(False)
        # Clear the saved data of the last forward.
        self.reset_outputs(self.student_outputs)
        self.reset_outputs(self.assist_outputs)
        #self.reset_outputs(self.student_grads)

        output = student(**data)
        if not self.assist:
            return output
        
        if not self.adjust_device:
            self.ta.to(output['acc'].device)
            self.adjust_device=True
        if self.training:
            self.ta_losses = self.ta.forward_train(self.student_feats[0][1:], data['img_metas'], data['gt_bboxes'], data['gt_labels']) 
        #import pdb;pdb.set_trace()
        #self.student_grad_hook(output)
        # print(data['img'].shape)
        # m = [k for k in self.student_outputs.keys()]
        # print(output, self.student_outputs[m[0]][0].shape)
        # grad = torch.autograd.grad(output['loss_cls'], self.student_outputs[m[0]][0], retain_graph=True)
        # print(grad[0].shape)
        # torch.save({"img": data['img'], "feat":self.student_outputs[m[0]][0], "grad": self.student_grads[m[0]]}, '/home/frank/Desktop/notebooks/student_tensors_%d.pt' % self.temp_count)
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
            # Get the teacher's outputs.
            teacher_module_name = component['teacher_module']
            teacher_outputs = self.get_teacher_outputs(teacher_module_name)
            # Get the student's outputs.
            if component.get('student_module') is not None:
                student_module_name = component['student_module']
                student_outputs = self.student_outputs[student_module_name]

                # Align student output's channels with teacher.
                align_module_name = f'component_{i}'
                if align_module_name in self.align_modules:
                    align_module = self.align_modules[align_module_name]
                    student_outputs = [
                        align_module(s_out) for s_out in student_outputs
                    ]
 
                for out_idx, (s_out, t_out) in enumerate(zip(student_outputs, teacher_outputs)):
                    for loss in component.losses:
                        loss_module = self.losses[loss.name]
                        loss_name = f'{loss.name}.{out_idx}'
                        # TODO ugly implementation.
                        # Pass the gt_label to loss function.
                        # Only used by WSLD.
                        loss_module.current_data = data
                        if loss.type == 'ChannelSpatialAttention':
                            # TODO: currently not consider retinanet, multi output/grads in one module
                            losses[loss_name] = loss_module(s_out, t_out, student_grad, teacher_grad)
                        else:
                            losses[loss_name] = loss_module(s_out, t_out)
                        loss_module.current_data = None

            if component.get('assist_module') is not None:
                assist_module_name = component['assist_module']
                assist_outputs = self.assist_outputs[assist_module_name]
                for out_idx, (a_out, t_out) in enumerate(zip(assist_outputs, teacher_outputs)):
                    for loss in component.losses:
                        loss_module = self.losses[loss.name]
                        loss_name = f'{loss.name}.{out_idx}'
                        # TODO ugly implementation.
                        # Pass the gt_label to loss function.
                        # Only used by WSLD.
                        loss_module.current_data = data
                        losses[loss_name] = loss_module(a_out, t_out)
                        loss_module.current_data = None

            #student_grad = self.student_grads[student_module_name]
            #teacher_grad = self.teacher_grads[teacher_module_name]
        if self.assist:
            for loss_k, loss_v in self.ta_losses.items():
                loss_v *= self.assist_loss_mul
                losses[loss_k] = loss_v
            #import pdb;pdb.set_trace()
        return losses
