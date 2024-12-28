import copy
import cv2 as cv
import mmcv
import numpy as np
import torch
import warnings
from mmcv.cnn import bias_init_with_prob, constant_init, xavier_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    TRANSFORMER_LAYER,
    TRANSFORMER_LAYER_SEQUENCE,
)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.base_module import BaseModule
from mmcv.utils import TORCH_VERSION, digit_version, ext_loader
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from re import T

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer

ext_module = ext_loader.load_ext(
    "_ext", ["ms_deform_attn_backward", "ms_deform_attn_forward"]
)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(
        self,
        *args,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        dataset_type="nuscenes",
        **kwargs,
    ):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(
        H,
        W,
        Z=8,
        num_points_in_pillar=4,
        dim="3d",
        bs=1,
        device="cuda",
        dtype=torch.float,
    ):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        
        if dim == "3d":
            zs = (
                torch.linspace(
                    0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device
                )
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        
        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    
    @force_fp32(apply_to=("reference_points", "img_metas"))
    def point_sampling(self, reference_points, pc_range, img_metas):

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )  

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(-1)
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )
        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(
        self,
        bev_query,
        key,
        value,
        *args,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=0.0,
        **kwargs,
    ):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )  
        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            dim="2d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )  

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs["img_metas"]
        )  

        
        
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(
                bs * 2, len_bev, -1
            )
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs,
            )

            bev_query = output  
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
        self,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

    def forward(
        self,
        query,
        key=None,
        value=None,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            
            if layer == "self_attn":

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


import torch.nn as nn
mask_supervison = True
gamma_weight = 1.0



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class HeightFormerEncoderT3SaveMenmoryMutiScale(BaseModule):

    """



    批次串行
    """

    def __init__(
        self,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        dataset_type="nuscenes",
        head=None,
        height_value=16,
    ):

        super(HeightFormerEncoderT3SaveMenmoryMutiScale, self).__init__()
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

        self.embedding = 256
        self.height_values = height_value
        self.head = 1  

        if head is not None:
            self.head = head




        height_probability = nn.Linear(
            self.embedding, self.height_values * self.head
        )  


        self.num_lvl=3
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        self.height_probability_list=_get_clones(height_probability,self.num_lvl) 
        self.loss_heght = nn.L1Loss()

        if mask_supervison:
            img_confidence = nn.Sequential(
                nn.Conv2d(self.embedding, self.embedding, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(self.embedding, 1, 1, 1, 0),
            )

            self.img_confidence_list=_get_clones(img_confidence,self.num_lvl)

            constant_init(self.img_confidence_list, 0.0)  

            bias_init = bias_init_with_prob(0.51)


            for i in range(self.num_lvl):
                nn.init.constant_(self.img_confidence_list[i][-1].bias, bias_init)

            self.loss_img_confidence = nn.L1Loss()  

    @staticmethod
    def get_reference_points(
        H,
        W,
        Z=8,
        num_points_in_pillar=1,
        dim="3d",
        bs=1,
        device="cuda",
        dtype=torch.float,
    ):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        
        if dim == "3d":
            zs = (
                torch.linspace(
                    0.001, Z - 0.001, num_points_in_pillar, dtype=dtype, device=device
                )
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )  
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        
        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    
    @force_fp32(apply_to=("reference_points", "img_metas"))
    def point_sampling(self, reference_points, pc_range, img_metas, heght_in_bev):

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        
        lidar2img = reference_points.new_tensor(lidar2img)  
        reference_points = reference_points.clone()
        
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )  
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )  

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(-1)
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]  
        reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]  

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )
        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    @force_fp32(apply_to=("reference_points", "img_metas"))
    def point_sampling_gt(self, reference_points, pc_range, img_metas, heght_in_bev):

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        height_ego = -img_meta["height_ego"]
        lidar2img = reference_points.new_tensor(lidar2img)  
        reference_points = reference_points.clone()
        
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )  
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        
        

        reference_points[..., 2:3] = reference_points[..., 2:3] * 3 + (
            pc_range[2] - 1
        )  

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(-1)
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][
            1
        ]  
        reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]  

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )
        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(
        self,
        bev_query,  
        key,  
        value,  
        *args,
        bev_h=None,  
        bev_w=None,  
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=0.0,
        heght_in_bev=None,
        height_mutiscale=False,
        index_lvl=0,
        **kwargs,
    ):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims) 1 6000 8 256 这里传进来的应该是原始的图像坐标
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        intermediate = []
        
        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            4,
            self.height_values,
            dim="3d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )  

        height_ego = -kwargs["img_metas"][0]["height_ego"]  
        

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs["img_metas"], height_ego
        )  

        

        img_shape = kwargs["img_metas"][0]["img_shape"][0]  
        img_h_high_level = img_shape[0] // 32  
        img_w_high_level = img_shape[1] // 32  
        if height_mutiscale:
            if kwargs["img_h"] is not None:  
                img_h_high_level = kwargs["img_h"]
                img_w_high_level = kwargs["img_w"]

        input = value.squeeze(1)  
        if mask_supervison:
            
            a, b, c, d = input.shape  
            input_temp = input.permute(0, 2, 1, 3).reshape(
                a * c, img_h_high_level, img_w_high_level, d
            )  

            input_temp = input_temp.permute(0, 3, 1, 2)  

            value_confidence = self.img_confidence_list[index_lvl](
                input_temp
            )  
            value_confidence = value_confidence.sigmoid()  
            input_mask = input_temp * value_confidence  
            input_temp = input_temp + input_mask  

            input = (
                input_temp.permute(0, 2, 3, 1)
                .reshape(a, c, img_h_high_level * img_w_high_level, d)
                .permute(0, 2, 1, 3)
            )  

            a, b, c, d = value_confidence.shape
            value_confidence = value_confidence.permute(0, 2, 3, 1)  
            reference_points_cam_0_1, bev_mask_0_1 = self.point_sampling_gt(
                ref_3d, self.pc_range, kwargs["img_metas"], height_ego
            )  
            reference_points_cam_0_1[
                ~(bev_mask_0_1[:, :, :, :, None].repeat(1, 1, 1, 1, 2))
            ] = 1.01  

        input_avg = input.mean(1)  

        num_cam, num_size_feature, bs, num_channel = input.shape
        reference_points_cam[
            ~(bev_mask[:, :, :, :, None].repeat(1, 1, 1, 1, 2))
        ] = 1.01  

        reference_points_cam = reference_points_cam * 2 - 1.0  
        

        D = reference_points_cam.size(3)  
        indexes = []
        
        for i, mask_per_img in enumerate(
            bev_mask
        ):  
            index_query_per_img = (
                mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            )  
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        reference_points_rebatch = (
            reference_points_cam.new_zeros([bs, num_cam, max_len, D, 2]) + 1.02
        )  

        for j in range(bs):  
            for i, reference_points_per_img in enumerate(reference_points_cam):  
                index_query_per_img = indexes[i]
                reference_points_rebatch[
                    j, i, : len(index_query_per_img)
                ] = reference_points_per_img[
                    j, index_query_per_img
                ]  

        input = input.permute(0, 2, 3, 1).permute(1, 0, 2, 3)  
        input = input.reshape(
            bs * num_cam, num_channel, img_h_high_level, img_w_high_level
        )  
        grid = reference_points_rebatch.reshape(
            bs * num_cam, max_len, self.height_values, 2
        )



        temp1 =  bev_query.permute(1, 0, 2)  
        heght_in_bev_pre_pro = self.height_probability_list[index_lvl](temp1)  
        a, b, c = heght_in_bev_pre_pro.shape
        heght_in_bev_pre_pro = heght_in_bev_pre_pro.reshape(
            a, b, c // self.head, self.head
        )  

        
        heght_in_bev_pre_pro = heght_in_bev_pre_pro.softmax(
            -2
        )  




    
        if mask_supervison:
            
            gt_label_sample = (
                reference_points_cam_0_1.clone()
                .detach()
                .reshape(num_cam, bs, bev_h * bev_w, self.height_values, 2)
            )  

            gt_label_sample_data = torch.zeros_like(value_confidence)  
            gt_label_confidence = (
                heght_in_bev_pre_pro.clone().detach().squeeze(-1)
            )  

            for i in range(num_cam):
                temp1 = gt_label_sample[i]  

                
                
                
                
                
                gt_label_sample[i] = temp1

            gt_label_sample = gt_label_sample.reshape(
                num_cam * bs, bev_h * bev_w * self.height_values, 2
            )  
            gt_label_sample = torch.clamp(gt_label_sample, 0, 0.99)  
            gt_label_sample[:, :, 0] = gt_label_sample[:, :, 0] * img_w_high_level  
            gt_label_sample[:, :, 1] = gt_label_sample[:, :, 1] * img_h_high_level  
            gt_label_sample = torch.floor(gt_label_sample).long()

            for i in range(num_cam * bs):  
                index = gt_label_sample[i, :, :]  
                gt_label_sample_data[i, index[:, 1], index[:, 0], :] = 1.0
            gt_label_sample_data = gt_label_sample_data.detach()
            

            loss_img_confidence = gamma_weight * self.loss_img_confidence(
                value_confidence.reshape(
                    num_cam * bs, img_h_high_level * img_w_high_level, 1
                ),
                gt_label_sample_data.reshape(
                    num_cam * bs, img_h_high_level * img_w_high_level, 1
                ),
            )  

            if kwargs is not None and kwargs.get("loss_dic") is not None:
                kwargs["loss_dic"]["loss_img_feature1"] = loss_img_confidence

        output = torch.zeros_like(bev_query).detach()  

        count = bev_mask.sum(-1) > 0  
        count = count.permute(1, 2, 0).sum(-1)  
        count = torch.clamp(count, min=1.0)  
        count = count[:, None, :, None]

        for index_o in range(bs):  
            slot = torch.zeros(
                [1, num_channel, bev_h * bev_w, self.height_values],
                dtype=bev_query.dtype,
                device=bev_query.device,
            )  
            temp_grid = grid[index_o * num_cam : (index_o * num_cam + num_cam), :, :, :]
            input_temp = input[
                index_o * num_cam : (index_o * num_cam + num_cam), :, :, :
            ]
            temp = torch.nn.functional.grid_sample(
                input_temp,
                temp_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=None,
            )  

            temp = temp.reshape(
                1, num_cam, num_channel, max_len, self.height_values
            )  
            for j in range(1):
                for i, index_query_per_img in enumerate(indexes):  
                    slot[0, :, index_query_per_img, :] += temp[
                        j, i, :, : len(index_query_per_img), :
                    ]  
            count_temp = count[index_o : (index_o + 1), :, :, :]
            slot = slot / count_temp  

            temp = slot

            temp = temp.permute(2, 0, 3, 1)  
            a, b, c, d = temp.shape
            temp = temp.reshape(
                a, c, d // self.head, self.head
            )  
            heght_in_bev_pre_pro_temp = heght_in_bev_pre_pro.permute(1, 0, 2, 3)[
                :, index_o, :, :
            ][
                :, :, None, :
            ]  
            temp = temp * heght_in_bev_pre_pro_temp  
            temp = temp.sum(dim=1)  
            output[:, index_o : (index_o + 1), :] = temp.reshape(a, b, d)

        output = output
        output = output.permute(1, 0, 2)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output





