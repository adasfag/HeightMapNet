import torch
import torch.nn.functional as F

from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxes[..., 1::2] = bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]

    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts


@BBOX_ASSIGNERS.register_module()
class AreaQueryAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(
        self,
        cls_cost=dict(type="ClassificationCost", weight=1.0),  # 分类损失
        reg_cost=dict(type="BBoxL1Cost", weight=1.0),
        iou_cost=dict(type="IoUCost", weight=0.0),
        pts_cost=dict(
            type="ChamferDistance", loss_src_weight=1.0, loss_dst_weight=1.0
        ),  # 点损失
        pc_range=None,
    ):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pts_cost = build_match_cost(pts_cost)
        self.pc_range = pc_range

    def assign(
        self,
        bbox_pred,
        cls_pred,
        pts_pred,
        gt_bboxes,
        gt_labels,
        gt_pts,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        assert bbox_pred.shape[-1] == 4, "Only support bbox pred shape is 4 dims"
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)  # 18 60

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)  # 60
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)  # 60
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return (
                AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels),
                None,
            )

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)  # 60 18 计算cls_cost矩阵
        # regression L1 cost

        # if len(gt_bboxes)==0:
        #     print('a')
        normalized_gt_bboxes = normalize_2d_bbox(
            gt_bboxes, self.pc_range
        )  # 18 4 对gt_bbox归一化
        # normalized_gt_bboxes = gt_bboxes
        # import pdb;pdb.set_trace()
        reg_cost = self.reg_cost(
            bbox_pred[:, :4], normalized_gt_bboxes[:, :4]
        )  # 60 18 计算bbox回归损失

        _, num_orders, num_pts_per_gtline, num_coords = gt_pts.shape
        normalized_gt_pts = normalize_2d_pts(gt_pts, self.pc_range)
        num_pts_per_predline = pts_pred.size(1)
        if num_pts_per_predline != num_pts_per_gtline:
            pts_pred_interpolated = F.interpolate(
                pts_pred.permute(0, 2, 1),
                size=(num_pts_per_gtline),
                mode="linear",
                align_corners=True,
            )
            pts_pred_interpolated = pts_pred_interpolated.permute(0, 2, 1).contiguous()
        else:
            pts_pred_interpolated = pts_pred
        # num_q, num_pts, 2 <-> num_gt, num_pts, 2
        pts_cost_ordered = self.pts_cost(
            pts_pred_interpolated, normalized_gt_pts
        )  # 60 162
        # normalized_gt_pts  9 9 10 2
        # pts_pred_interpolated 60 10 2
        centre_nnormalized_gt_pts = normalized_gt_pts.mean(
            -2, keepdim=True
        )  # 9 9 1 2 中心点

        pts_cost_ordered = pts_cost_ordered.view(
            num_bboxes, num_gts, num_orders
        )  # 60 18 9
        pts_cost, order_index = torch.min(
            pts_cost_ordered, 2
        )  # pts_cost 60 18 order_index 60 18

        # bboxes = denormalize_2d_bbox(bbox_pred, self.pc_range)
        # iou_cost = self.iou_cost(bboxes, gt_bboxes)#全为0
        # # weighted sum of above three costs
        # cost = cls_cost + reg_cost + iou_cost + pts_cost#60 18

        # #这里cost可能会报一个matrix contains invalid numeric entries 错误
        # #print(cost) qwz 原因未找到

        # isnotnun=torch.isfinite(cost)
        # if torch.sum(isnotnun)>=0:
        #     #print("\ncost becomes a num{}\n".format(torch.sum(isnotnun)))
        #     pass
        # cost[~isnotnun]=10000.0#qwz

        # # 3. do Hungarian matching on CPU using linear_sum_assignment
        # cost = cost.detach().cpu()
        # if linear_sum_assignment is None:
        #     raise ImportError('Please run "pip install scipy" '
        #                       'to install scipy first.')
        # matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        # #cost 60 18 -> matched_row_inds 18 （矩阵行坐标）  matched_col_inds 18 (矩阵列坐标)
        # matched_row_inds = torch.from_numpy(matched_row_inds).to(
        #     bbox_pred.device)
        # matched_col_inds = torch.from_numpy(matched_col_inds).to(
        #     bbox_pred.device)

        # # 4. assign backgrounds and foregrounds
        # # assign all indices to backgrounds first
        # assigned_gt_inds[:] = 0#全部为0 认为0是背景
        # assign foregrounds based on matching results
        x = torch.floor(
            centre_nnormalized_gt_pts[:, 0, 0, 0] * 10
        ).long()  # 网格x坐标9 test 应该不受影响
        y = torch.floor(
            (centre_nnormalized_gt_pts[:, 0, 0, 1]) * (num_bboxes // 10)
        ).long()  # 网格y坐标 9
        matched_row_inds = x + y * 10  # 有可能多对一 相近的元素
        matched_row_inds = x + y * 10  # 有可能多对一 相近的元素
        temp = matched_row_inds.clone()
        count = 0
        unique_index = matched_row_inds.unique()
        while len(unique_index) != len(matched_row_inds):
            if count > 100:
                print(
                    "the number of overlap is",
                    (len(matched_row_inds)) - len(unique_index),
                )
                break
            if count > 10:
                count = count + 1
                unique_index = matched_row_inds.unique()  # 缓解多对一问题但是不能解决这个问题
                for i in unique_index:
                    data = matched_row_inds[matched_row_inds == i]
                    if len(data) >= 2:  # 有重复问题
                        matched_row_inds[matched_row_inds == i] = data[0] + torch.range(
                            -(len(data)) // 2,
                            -(len(data)) // 2 + len(data) - 1,
                            dtype=torch.long,
                            device=gt_labels.device,
                        )
            count = count + 1
            unique_index = matched_row_inds.unique()  # 缓解多对一问题但是不能解决这个问题
            for i in unique_index:
                data = matched_row_inds[matched_row_inds == i]
                if len(data) >= 2:  # 有重复问题
                    matched_row_inds[matched_row_inds == i] = (
                        data[0]
                        + torch.range(
                            (3 - data[0] // 10) - (len(data)) // 2,
                            (3 - data[0] // 10) - (len(data)) // 2 + len(data) - 1,
                            dtype=torch.long,
                            device=gt_labels.device,
                        )
                        * 10
                    )
            matched_row_inds[matched_row_inds < 0] = 0
            matched_row_inds[matched_row_inds > (len(pts_cost_ordered) - 1)] = (
                len(pts_cost_ordered) - 1
            )

        matched_row_inds[matched_row_inds < 0] = 0
        matched_row_inds[matched_row_inds > (len(pts_cost_ordered) - 1)] = (
            len(pts_cost_ordered) - 1
        )

        matched_col_inds = torch.range(
            0, len(matched_row_inds) - 1, dtype=torch.long, device=gt_labels.device
        )

        matched_pre = pts_pred_interpolated[matched_row_inds]  # 9 10 2
        pts_cost_ordered = self.pts_cost(matched_pre, normalized_gt_pts)  # 9 9 9
        pts_cost_ordered = pts_cost_ordered.view(
            len(matched_row_inds), num_gts, num_orders
        )  # 9 9 9
        pts_cost, order_index_temp = torch.min(pts_cost_ordered, 2)  # 9 9

        order_index = torch.zeros_like(order_index, dtype=order_index.dtype)
        assigned_gt_inds[matched_row_inds] = (
            matched_col_inds + 1
        )  # 60 每个预测对应的gt标签序号  ，每个标签序号多加1 0代表没有对应序号
        order_index[matched_row_inds] = order_index_temp

        assigned_gt_inds[matched_row_inds] = (
            matched_col_inds + 1
        )  # 60 每个预测对应的gt标签序号  ，每个标签序号多加1 0代表没有对应序号
        assigned_labels[matched_row_inds] = gt_labels[
            matched_col_inds
        ]  # 60 分配gt标签 这个是对的
        return (
            AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels),
            order_index,
        )


def getIdx(a):
    co = a.unsqueeze(0) - a.unsqueeze(1)
    uniquer = co.unique(dim=0)
    out = []
    for r in uniquer:
        cover = torch.arange(a.size(0))
        mask = r == 0
        idx = cover[mask]
        out.append(idx)
    return out
