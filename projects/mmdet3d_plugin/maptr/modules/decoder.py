import torch
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from tkinter import N

from mmdet.models.utils.transformer import inverse_sigmoid


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapTRDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        reg_branches=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2
            )  # BS NUM_QUERY NUM_LEVEL 2 归一化坐标
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2]
                )
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


import torch.nn as nn


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoderFusion(TransformerLayerSequence):
    """
    采用编码器解码器架构合并最后一层 Decoder
    历史和未来同等对待
    并且要在transfomer那一层取出初始 query
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapTRDecoderFusion, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

        # 相关参数
        self.num_instance = 50
        self.num_point = 20
        self.num_sequees = 2
        self.encoder = nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims))
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims),
            nn.BatchNorm1d(self.embed_dims),  # N C
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        reg_branches=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2
            )  # BS NUM_QUERY NUM_LEVEL 2 归一化坐标
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            # 时序融合算法
            if lid == self.num_layers - 1:
                reference_points = reference_points[
                    (self.num_sequees - 1) :: self.num_sequees
                ]
                idetify = output  # 8 1000 256
                idetify = idetify[
                    (self.num_sequees - 1) :: self.num_sequees
                ]  # 靠前的那个是当前帧 4 1000 256

                output = self.encoder(output)  # 8 1000 256
                bs, num_query, num_channel = output.shape

                output = output.reshape(bs, self.num_instance, self.num_point, -1)
                output_max, _ = output.max(2, keepdim=True)
                output_max = output_max.repeat(1, 1, self.num_point, 1)
                output = torch.cat([output, output_max], dim=-1)

                output = output.reshape(
                    bs // self.num_sequees, self.num_sequees, num_query, -1
                )

                output = output.mean(1)  # 4 10000 256

                bs, num_query, _ = output.shape
                output = self.decoder(output.reshape(bs * num_query, -1))
                output = output.reshape(bs, num_query, num_channel)

                output = idetify + output

                # 时序融合算法

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2]
                )
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output_temp = output
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                if lid < self.num_layers - 1:
                    output_temp = output_temp[
                        (self.num_sequees - 1) :: self.num_sequees
                    ]
                    output_temp = output_temp.permute(1, 0, 2)
                    intermediate.append(output_temp)
                    intermediate_reference_points.append(
                        reference_points[(self.num_sequees - 1) :: self.num_sequees]
                    )
                else:
                    intermediate.append(output)
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


import copy
from mmcv.cnn import Linear, bias_init_with_prob, constant_init, xavier_init


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoderFusionV2(TransformerLayerSequence):
    """
    每一层都同等对待了
    只将之前的特征加到当前帧
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapTRDecoderFusionV2, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

        # 相关参数
        self.num_instance = 50
        self.num_point = 20
        self.num_sequees = 2
        self.num_reg_fcs = 2
        self.num_pred = 6
        self.code_size = 2

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # encoder是不是应该多层？
        encoder = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.BatchNorm1d(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.BatchNorm1d(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )  # 非常简单的encode

        fusion = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )

        decoder = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.BatchNorm1d(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.BatchNorm1d(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.encoder = _get_clones(encoder, self.num_pred)
        self.fusion = _get_clones(fusion, self.num_pred)
        self.decoder = _get_clones(decoder, self.num_pred)

        reg_branch_xiuzheng = []
        for _ in range(self.num_reg_fcs):  # 融合后特征可能经过后和原来的属于不同的修正维度
            reg_branch_xiuzheng.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch_xiuzheng.append(nn.ReLU())
        reg_branch_xiuzheng.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch_xiuzheng = nn.Sequential(*reg_branch_xiuzheng)

        self.reg_branch_xiuzheng = _get_clones(reg_branch_xiuzheng, self.num_pred)

        for m in self.reg_branch_xiuzheng:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branch_xiuzheng[1][-1].bias.data[2:], 0.0)

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        reg_branches=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        can_bus = kwargs["can_bus_embedding"]  # 1 8 256
        canbus = can_bus.permute(1, 0, 2)  # 8 1 256

        reference_points1 = reference_points  # 8 1000 2

        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points1[..., :2].unsqueeze(
                2
            )  # 8 1000 1 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )  # 1000 8 256
            output = output.permute(1, 0, 2)  # 8 1000 256

            # 时序融合算法
            # 所有层都是一样，在原来的基础上加上修正，编码之前加上位姿编码，解码之前也加上位姿编码
            idetify = output  # 8 1000 256
            idetify = idetify[
                (self.num_sequees - 1) :: self.num_sequees
            ]  # 靠前的那个是当前帧 4 1000 256

            bs, num_query, num_channel = output.shape
            output_fusion = self.encoder[lid](
                (output + canbus).reshape(bs * num_query, -1)
            )  # 8 1000 256
            output_fusion = output_fusion.reshape(bs, num_query, num_channel)
            bs, num_query, num_channel = output_fusion.shape

            output_fusion = output_fusion.reshape(
                bs, self.num_instance, self.num_point, -1
            )
            output_max, _ = output_fusion.max(2, keepdim=True)
            output_max = output_max.repeat(1, 1, self.num_point, 1)
            output_fusion = torch.cat([output_fusion, output_max], dim=-1)

            output_fusion = output_fusion.reshape(
                bs // self.num_sequees, self.num_sequees, num_query, -1
            )

            output_fusion = output_fusion.mean(1)  # 4 10000 256

            bs, num_query, _ = output_fusion.shape

            output_fusion = self.fusion[lid](output_fusion)

            output_fusion = self.decoder[lid](
                (
                    output_fusion + canbus[(self.num_sequees - 1) :: self.num_sequees]
                ).reshape(bs * num_query, -1)
            )
            output_fusion = output_fusion.reshape(bs, num_query, num_channel)

            output_fusion = idetify + output_fusion

            # 时序融合算法

            if reg_branches is not None:  # 在迭代的过程中用的是各自的，但是输出的时候用的是修正的
                # if lid <self.num_layers-1:
                #     output[(self.num_sequees-1)::self.num_sequees]=output_fusion#将融合的加上之前
                # else:
                #     output=output_fusion

                tmp = reg_branches[lid](output)  # 采样的特征
                temp1 = reg_branches[lid](output_fusion)  # 输出的特征

                assert reference_points1.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points1)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points1[..., :2]
                )

                new_reference_points_out = (
                    temp1[..., :2]
                    + new_reference_points[..., :2][
                        (self.num_sequees - 1) :: self.num_sequees
                    ]
                )  # 大家都是在反sigmoid 后 每一层采样和输出不同 输出的时候会进行修正

                new_reference_points[..., :2][
                    (self.num_sequees - 1) :: self.num_sequees
                ] = new_reference_points_out  # 采样的和优化的是同一个

                new_reference_points_out = new_reference_points_out.sigmoid()
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = (
                    new_reference_points.sigmoid()
                )  # 采样所依据的初始位置和输出的初始位置不一样
                reference_points1 = new_reference_points.detach()  # 第0层之后

            output_temp = output_fusion  # 输出的是out fusion
            output_fusion = output_fusion.permute(1, 0, 2)
            output = output.permute(1, 0, 2)  # 输入的是8批次的output
            if self.return_intermediate:
                if lid < self.num_layers - 1:
                    output_temp = output_fusion
                    intermediate.append(output_temp)
                    intermediate_reference_points.append(
                        new_reference_points_out
                    )  # 输出的是修正后的结果
                else:
                    output_temp = output_fusion
                    intermediate.append(output_temp)
                    intermediate_reference_points.append(new_reference_points_out)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points
            )  # 只输出当前帧

        return output, reference_points


import torch


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoderGeo(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapTRDecoderGeo, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        reg_branches=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2
            )  # BS NUM_QUERY NUM_LEVEL 2 归一化坐标
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                output = output.permute(1, 0, 2)
                num_query, bs, num_channel = output.shape  # 8 1000 256

                num_instance = 50
                num_point = 20

                query_temp = output.reshape(num_instance, num_point, bs, num_channel)

                query_max, _ = query_temp.max(dim=1, keepdim=True)

                query_avg = query_max.mean(0, keepdim=True)

                query_max = query_max.repeat(1, num_point, 1, 1)

                query_avg = query_avg.repeat(num_instance, num_point, 1, 1)

                query = torch.cat([query_temp, query_max, query_avg], dim=-1)

                output = query.reshape(num_query, bs, 3 * num_channel)
                output = output.permute(1, 0, 2)  # 8 1000 768

                tmp = reg_branches[lid](output)
                output = output[:, :, :num_channel]

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2]
                )
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class AreaQueryDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(AreaQueryDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        reg_branches=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2
            )  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2]
                )
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence


@TRANSFORMER_LAYER.register_module()
class DecoupledDetrTransformerDecoderLayer(BaseTransformerLayer):
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
        num_vec=50,
        num_pts_per_vec=20,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(DecoupledDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )
        assert len(operation_order) == 8
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
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
        #
        num_vec = self.num_vec
        num_pts_per_vec = self.num_pts_per_vec

        self_attn_mask = None
        # self_attn_mask = (
        #     torch.zeros([num_vec, num_vec,]).bool().to(query.device)
        # )
        # self_attn_mask[ num_vec :, 0 :  num_vec,] = True
        # self_attn_mask[0 :  num_vec,  num_vec :,] = True#这个在一对多的时候也许有用
        for layer in self.operation_order:
            if layer == "self_attn":
                # import ipdb;ipdb.set_trace()
                if attn_index == 0:
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(
                        num_vec, num_pts_per_vec, n_batch, n_dim
                    ).flatten(1, 2)
                    query_pos = query_pos.view(
                        num_vec, num_pts_per_vec, n_batch, n_dim
                    ).flatten(
                        1, 2
                    )  # 50 n_batch*20 256
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=self_attn_mask,
                        key_padding_mask=query_key_padding_mask,
                        **kwargs,
                    )
                    # import ipdb;ipdb.set_trace()
                    query = query.view(
                        num_vec, num_pts_per_vec, n_batch, n_dim
                    ).flatten(0, 1)
                    query_pos = query_pos.view(
                        num_vec, num_pts_per_vec, n_batch, n_dim
                    ).flatten(0, 1)
                    attn_index += 1
                    identity = query
                else:
                    # import ipdb;ipdb.set_trace()
                    n_pts, n_batch, n_dim = query.shape
                    query = (
                        query.view(num_vec, num_pts_per_vec, n_batch, n_dim)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                        .flatten(1, 2)
                    )  # 20 50 256
                    query_pos = (
                        query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                        .flatten(1, 2)
                    )
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs,
                    )
                    # import ipdb;ipdb.set_trace()
                    query = (
                        query.view(num_pts_per_vec, num_vec, n_batch, n_dim)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                        .flatten(0, 1)
                    )
                    query_pos = (
                        query_pos.view(num_pts_per_vec, num_vec, n_batch, n_dim)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                        .flatten(0, 1)
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
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
    