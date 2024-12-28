# 将分类损失对齐到回归损失
from .nms_free_coder import MapTRNMSFreeCoder, MapTRNMSFreeCoderV03, NMSFreeCoder

__all__ = ["NMSFreeCoder", "MapTRNMSFreeCoder"]
