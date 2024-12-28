from __future__ import absolute_import, division, print_function

import GeometricKernelAttention as GKA
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class GeometricKernelAttentionFunc(Function):
    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        bs, _, _, _ = value.shape
        if bs > im2col_step:  # qwz训练和测试这个数可能不一样
            im2col_step = im2col_step * 2
            # if bs >im2col_step: #qwz训练和测试这个数可能不一样
            #     im2col_step=im2col_step*2
        # im2col_step=32
        ctx.im2col_step = im2col_step
        output = GKA.geometric_kernel_attn_cuda_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_attn_weight = GKA.geometric_kernel_attn_cuda_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, None, grad_attn_weight, None
