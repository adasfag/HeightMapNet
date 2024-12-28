import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init, xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from projects.mmdet3d_plugin.bevformer.modules.decoder import (
    CustomMSDeformableAttention,
)
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import (
    MSDeformableAttention3D,
)
from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import (
    TemporalSelfAttention,
)
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate
from typing import List

from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid
from .builder import FUSERS, build_fuser


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import (
    build_attention,
    build_feedforward_network,
    build_positional_encoding,
)

@TRANSFORMER.register_module()
class MapTRPerceptionTransformerAddBevPosAddMutiScale(BaseModule):
    """
    增加多尺度融合方案
    """

    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        fuser=None,
        encoder=None,
        decoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        modality="vision",
        attn_cfgs=None,
        norm_cfg=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=256 * 2,
            num_fcs=2,
            ffn_dropout=0.1,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        **kwargs
    ):
        super(MapTRPerceptionTransformerAddBevPosAddMutiScale, self).__init__(**kwargs)
        if modality == "fusion":
            self.fuser = build_fuser(fuser)  
        self.use_attn_bev = True
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

        self.feature_levels = 1
        reduce_muti_scale_cnn = nn.Sequential(
            nn.Conv2d(
                self.embed_dims * self.feature_levels,
                self.embed_dims,
                5,  
                1,
                2,
            ),
            nn.BatchNorm2d(self.embed_dims),
            nn.ReLU(),
            nn.Conv2d(
                self.embed_dims,
                self.embed_dims,
                5,  
                1,
                2,
            ),
            nn.BatchNorm2d(self.embed_dims),
            nn.ReLU(),
        )






        self.feature_levels_muti_scale = 3
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


        self.reduce_muti_scale_cnn_list=_get_clones(reduce_muti_scale_cnn,   self.feature_levels_muti_scale)


        increase_per_feild = nn.Conv2d(
            self.embed_dims * self.feature_levels,
            self.embed_dims,
            5,  
            1,
            2,
        )

        self.increase_per_feild_list=_get_clones(increase_per_feild,   self.feature_levels_muti_scale)

        self.norm = build_norm_layer(dict(type="LN"), self.embed_dims)[1]
        self.ffns = build_feedforward_network(
            dict(
                type="FFN",
                embed_dims=256,
                feedforward_channels=256 * 2,
                num_fcs=2,
                ffn_dropout=0.1,
                act_cfg=dict(type="ReLU", inplace=True),
            )
        )

        self.attention_config = attn_cfgs
    

        self.reduce_muti_scale_cnn_mutiscale = nn.Conv2d(
            self.embed_dims * self.feature_levels_muti_scale,
            self.embed_dims,
            5,  
            1,
            2,
        )

        if self.attention_config is not None:

            self.attention_fusion = build_attention(self.attention_config)
            self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
            self.ffns = build_feedforward_network(ffn_cfgs)

            self.positional_encoding = build_positional_encoding(positional_encoding)

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2)  
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module("norm", nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if (
                isinstance(m, MSDeformableAttention3D)
                or isinstance(m, TemporalSelfAttention)
                or isinstance(m, CustomMSDeformableAttention)
            ):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution="uniform", bias=0.0)
        xavier_init(self.can_bus_mlp, distribution="uniform", bias=0.0)

    
    
    def attn_bev_encode(
        self,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs
    ):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        
        delta_x = np.array([each["can_bus"][0] for each in kwargs["img_metas"]])
        delta_y = np.array([each["can_bus"][1] for each in kwargs["img_metas"]])
        ego_angle = np.array(
            [each["can_bus"][-2] / np.pi * 180 for each in kwargs["img_metas"]]
        )
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = (
            translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        )
        shift_x = (
            translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        )
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(
            1, 0
        )  

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    
                    rotation_angle = kwargs["img_metas"][i]["can_bus"][-1]
                    tmp_prev_bev = (
                        prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    )
                    tmp_prev_bev = rotate(
                        tmp_prev_bev, rotation_angle, center=self.rotate_center
                    )
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1
                    )
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        
        can_bus = bev_queries.new_tensor(
            [each["can_bus"] for each in kwargs["img_metas"]]
        )  
        bs, num_channel = can_bus.shape
        if num_channel != 18:
            can_bus = can_bus.reshape(bs * (num_channel) // 18, 18)
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries

        

        
        bev_embed_s_muti_scale = []



        


        ref_3d = self.encoder.get_reference_points(
            bev_h,
            bev_w,
            4,
            self.encoder.height_values,
            dim="3d",
            bs=bev_queries.size(1),
            device=bev_queries.device,
            dtype=bev_queries.dtype,
        )  

        

        reference_points_cam, bev_mask = self.encoder.point_sampling(
            ref_3d, self.encoder.pc_range, kwargs["img_metas"], -1.8
        )  

        temp_count = bev_mask.sum(-1) > 0  
        num_cam, bs, num_grids = temp_count.shape


        input_avg=mlvl_feats[-1].mean(-1).mean(-1).permute(1,0,2)

        input_avg = input_avg[:, :, None, :].repeat(1, 1, num_grids, 1) 
        input_avg[~temp_count] = 0.0
        temp_count = temp_count.permute(1, 2, 0).sum(-1)  
        temp_count = torch.clamp(temp_count, min=1.0)  

        input_avg = torch.sum(input_avg, dim=0) / temp_count[:, :, None]  



        bev_queries = bev_queries.permute(1, 0, 2)
        bev_embed_s = bev_queries.reshape(bs, bev_h, bev_w, self.embed_dims).permute(
            0, 3, 1, 2
        )  

        bev_embed_s = self.reduce_muti_scale_cnn_list[2](bev_embed_s)  
        bev_queries=bev_embed_s.reshape(bs,self.embed_dims,bev_h*bev_w).permute(0, 2, 1)

        bev_queries = input_avg + bev_queries  

        bev_queries=bev_queries.permute(1,0,2)

        for index_lvl in range(len(mlvl_feats)):  
            feat_flatten = []
            spatial_shapes = []
            for lvl, feat in enumerate(mlvl_feats[index_lvl][None]):  
                bs, num_cam, c, h, w = feat.shape
                spatial_shape = (h, w)
                feat = feat.flatten(3).permute(1, 0, 3, 2)
                if self.use_cams_embeds:
                    feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
                feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(
                    feat.dtype
                )
                spatial_shapes.append(spatial_shape)
                feat_flatten.append(feat)

            feat_flatten = torch.cat(feat_flatten, 2)
            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=bev_pos.device
            )
            level_start_index = torch.cat(
                (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
            )

            feat_flatten = feat_flatten.permute(
                0, 2, 1, 3
            )  



            if index_lvl<=1:
                bev_queries_index=bev_queries.clone()
            else:
                bev_queries_index=bev_queries
            bev_embed = self.encoder(
                bev_queries_index,  
                feat_flatten,  
                feat_flatten,  
                bev_h=bev_h,  
                bev_w=bev_w,  
                bev_pos=None,  
                spatial_shapes=spatial_shapes,  
                level_start_index=level_start_index,  
                prev_bev=prev_bev,
                shift=shift,  
                img_h=h,
                img_w=w,
                height_mutiscale=True,
                index_lvl=index_lvl,
                **kwargs
            )  
            bev_embed = bev_embed + bev_queries.permute(1, 0, 2)
            bev_embed = self.norm(bev_embed)
            bev_embed = self.ffns(bev_embed)
            bev_embed = self.norm(bev_embed)
            bev_embed = bev_embed

            
            
            
            
            
            
            bev_embed_s_muti_scale.append(bev_embed)

        bev_embed_s = torch.cat(bev_embed_s_muti_scale, dim=2)  

        
        bev_embed_s = bev_embed_s.reshape(
            bs, bev_h, bev_w, self.embed_dims * 3
        ).permute(
            0, 3, 1, 2
        )  

        bev_embed_s = self.reduce_muti_scale_cnn_mutiscale(
            bev_embed_s
        )  

        bev_embed_q = bev_embed_s.permute(0, 2, 3, 1).reshape(
            bs, bev_h * bev_w, self.embed_dims
        )  

        
        dtype = torch.float
        device = bev_embed_q.device
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=dtype, device=device),
            torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=dtype, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / bev_h
        ref_x = ref_x.reshape(-1)[None] / bev_w
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)  

        bev_fature_mask = torch.zeros((bs, bev_h, bev_w), device=bev_queries.device).to(
            dtype
        )
        bev_fature_pos = self.positional_encoding(bev_fature_mask).to(
            dtype
        )  
        bev_fature_pos = (
            bev_fature_pos.flatten(2).permute(2, 0, 1).permute(1, 0, 2)
        )  

        query = self.attention_fusion(
            bev_embed_q,
            bev_embed_q,
            bev_embed_q,
            query_pos=bev_fature_pos,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=device),
            level_start_index=torch.tensor([0], device=bev_embed_q.device),
        )  
        query = self.norm(query)
        query = self.ffns(query)
        query = self.norm(query)

        return query, can_bus

    def lss_bev_encode(self, mlvl_feats, prev_bev=None, **kwargs):
        assert (
            len(mlvl_feats) == 1
        ), "Currently we only support single level feat in LSS"
        images = mlvl_feats[0]
        img_metas = kwargs["img_metas"]
        bev_embed = self.encoder(images, img_metas)
        bs, c, _, _ = bev_embed.shape
        bev_embed = bev_embed.view(bs, c, -1).permute(0, 2, 1).contiguous()

        return bev_embed

    def get_bev_features(
        self,
        mlvl_feats,
        lidar_feat,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs
    ):
        """
        obtain bev features.
        """
        can_bus = None
        if self.use_attn_bev:
            bev_embed, can_bus = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs
            )
        else:
            bev_embed = self.lss_bev_encode(mlvl_feats, prev_bev=prev_bev, **kwargs)
        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = (
                bev_embed.view(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()
            )
            lidar_feat = lidar_feat.permute(0, 1, 3, 2).contiguous()  
            lidar_feat = nn.functional.interpolate(
                lidar_feat, size=(bev_h, bev_w), mode="bicubic", align_corners=False
            )
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0, 2, 1).contiguous()
            bev_embed = fused_bev
        if can_bus is not None:
            return bev_embed, can_bus
        return bev_embed

    
    
    def forward(
        self,
        mlvl_feats,
        lidar_feat,
        bev_queries,
        object_query_embed,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        prev_bev=None,
        **kwargs
    ):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        kwargs["can_bus_embedding"] = None
        bev_embed = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs
        )  
        if isinstance(bev_embed, tuple):
            bev_embed, canbus = bev_embed

        kwargs["can_bus_embedding"] = canbus
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs
        )

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
    


