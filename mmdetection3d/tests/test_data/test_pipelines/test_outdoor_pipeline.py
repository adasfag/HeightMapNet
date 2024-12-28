# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose


def test_outdoor_aug_pipeline():
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    class_names = ["Car"]
    np.random.seed(0)

    train_pipeline = [
        dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
        dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
        dict(
            type="ObjectNoise",
            num_try=100,
            translation_std=[1.0, 1.0, 0.5],
            global_rot_range=[0.0, 0.0],
            rot_range=[-0.78539816, 0.78539816],
        ),
        dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
        dict(
            type="GlobalRotScaleTrans",
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05],
        ),
        dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
        dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
        dict(type="PointShuffle"),
        dict(type="DefaultFormatBundle3D", class_names=class_names),
        dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
    ]
    pipeline = Compose(train_pipeline)

    gt_bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [
                [
                    2.16902428e01,
                    -4.06038128e-02,
                    -1.61906636e00,
                    1.65999997e00,
                    3.20000005e00,
                    1.61000001e00,
                    -1.53999996e00,
                ],
                [
                    7.05006886e00,
                    -6.57459593e00,
                    -1.60107934e00,
                    2.27999997e00,
                    1.27799997e01,
                    3.66000009e00,
                    1.54999995e00,
                ],
                [
                    2.24698811e01,
                    -6.69203758e00,
                    -1.50118136e00,
                    2.31999993e00,
                    1.47299995e01,
                    3.64000010e00,
                    1.59000003e00,
                ],
                [
                    3.48291969e01,
                    -7.09058380e00,
                    -1.36622977e00,
                    2.31999993e00,
                    1.00400000e01,
                    3.60999990e00,
                    1.61000001e00,
                ],
                [
                    4.62394600e01,
                    -7.75838804e00,
                    -1.32405007e00,
                    2.33999991e00,
                    1.28299999e01,
                    3.63000011e00,
                    1.63999999e00,
                ],
                [
                    2.82966995e01,
                    -5.55755794e-01,
                    -1.30332506e00,
                    1.47000003e00,
                    2.23000002e00,
                    1.48000002e00,
                    -1.57000005e00,
                ],
                [
                    2.66690197e01,
                    2.18230209e01,
                    -1.73605704e00,
                    1.55999994e00,
                    3.48000002e00,
                    1.39999998e00,
                    -1.69000006e00,
                ],
                [
                    3.13197803e01,
                    8.16214371e00,
                    -1.62177873e00,
                    1.74000001e00,
                    3.76999998e00,
                    1.48000002e00,
                    2.78999996e00,
                ],
                [
                    4.34395561e01,
                    -1.95209332e01,
                    -1.20757008e00,
                    1.69000006e00,
                    4.09999990e00,
                    1.40999997e00,
                    -1.53999996e00,
                ],
                [
                    3.29882965e01,
                    -3.79360509e00,
                    -1.69245458e00,
                    1.74000001e00,
                    4.09000015e00,
                    1.49000001e00,
                    -1.52999997e00,
                ],
                [
                    3.85469360e01,
                    8.35060215e00,
                    -1.31423414e00,
                    1.59000003e00,
                    4.28000021e00,
                    1.45000005e00,
                    1.73000002e00,
                ],
                [
                    2.22492104e01,
                    -1.13536005e01,
                    -1.38272512e00,
                    1.62000000e00,
                    3.55999994e00,
                    1.71000004e00,
                    2.48000002e00,
                ],
                [
                    3.36115799e01,
                    -1.97708054e01,
                    -4.92827654e-01,
                    1.64999998e00,
                    3.54999995e00,
                    1.79999995e00,
                    -1.57000005e00,
                ],
                [
                    9.85029602e00,
                    -1.51294518e00,
                    -1.66834795e00,
                    1.59000003e00,
                    3.17000008e00,
                    1.38999999e00,
                    -8.39999974e-01,
                ],
            ],
            dtype=torch.float32,
        )
    )
    gt_labels_3d = np.array([0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    results = dict(
        pts_filename="tests/data/kitti/a.bin",
        ann_info=dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d),
        bbox3d_fields=[],
        img_fields=[],
    )

    output = pipeline(results)

    expected_tensor = torch.tensor(
        [
            [20.6514, -8.8250, -1.0816, 1.5893, 3.0637, 1.5414, -1.9216],
            [7.9374, 4.9457, -1.2008, 2.1829, 12.2357, 3.5041, 1.6629],
            [20.8115, -2.0273, -1.8893, 2.2212, 14.1026, 3.4850, 2.6513],
            [32.3850, -5.2135, -1.1321, 2.2212, 9.6124, 3.4562, 2.6498],
            [43.7022, -7.8316, -0.5090, 2.2403, 12.2836, 3.4754, 2.0146],
            [25.3300, -9.6670, -1.0855, 1.4074, 2.1350, 1.4170, -0.7141],
            [16.5414, -29.0583, -0.9768, 1.4936, 3.3318, 1.3404, -0.7153],
            [24.6548, -18.9226, -1.3567, 1.6659, 3.6094, 1.4170, 1.3970],
            [45.8403, 1.8183, -1.1626, 1.6180, 3.9254, 1.3499, -0.6886],
            [30.6288, -8.4497, -1.4881, 1.6659, 3.9158, 1.4265, -0.7241],
            [32.3316, -22.4611, -1.3131, 1.5223, 4.0977, 1.3882, 2.4186],
            [22.4492, 3.2944, -2.1674, 1.5510, 3.4084, 1.6372, 0.3928],
            [37.3824, 5.0472, -0.6579, 1.5797, 3.3988, 1.7233, -1.4862],
            [8.9259, -1.2578, -1.6081, 1.5223, 3.0350, 1.3308, -1.7212],
        ]
    )
    assert torch.allclose(
        output["gt_bboxes_3d"]._data.tensor, expected_tensor, atol=1e-3
    )


def test_outdoor_velocity_aug_pipeline():
    point_cloud_range = [-50, -50, -5, 50, 50, 3]
    class_names = [
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    ]
    np.random.seed(0)

    train_pipeline = [
        dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
        dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
        dict(
            type="GlobalRotScaleTrans",
            rot_range=[-0.3925, 0.3925],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0, 0, 0],
        ),
        dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
        dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
        dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
        dict(type="PointShuffle"),
        dict(type="DefaultFormatBundle3D", class_names=class_names),
        dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
    ]
    pipeline = Compose(train_pipeline)

    gt_bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [
                [
                    -5.2422e00,
                    4.0021e01,
                    -4.7643e-01,
                    2.0620e00,
                    4.4090e00,
                    1.5480e00,
                    -1.4880e00,
                    8.5338e-03,
                    4.4934e-02,
                ],
                [
                    -2.6675e01,
                    5.5950e00,
                    -1.3053e00,
                    3.4300e-01,
                    4.5800e-01,
                    7.8200e-01,
                    -4.6276e00,
                    -4.3284e-04,
                    -1.8543e-03,
                ],
                [
                    -5.8098e00,
                    3.5409e01,
                    -6.6511e-01,
                    2.3960e00,
                    3.9690e00,
                    1.7320e00,
                    -4.6520e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    -3.1309e01,
                    1.0901e00,
                    -1.0561e00,
                    1.9440e00,
                    3.8570e00,
                    1.7230e00,
                    -2.8143e00,
                    -2.7606e-02,
                    -8.0573e-02,
                ],
                [
                    -4.5642e01,
                    2.0136e01,
                    -2.4681e-02,
                    1.9870e00,
                    4.4400e00,
                    1.9420e00,
                    2.8336e-01,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    -5.1617e00,
                    1.8305e01,
                    -1.0879e00,
                    2.3230e00,
                    4.8510e00,
                    1.3710e00,
                    -1.5803e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    -2.5285e01,
                    4.1442e00,
                    -1.2713e00,
                    1.7550e00,
                    1.9890e00,
                    2.2200e00,
                    -4.4900e00,
                    -3.1784e-02,
                    -1.5291e-01,
                ],
                [
                    -2.2611e00,
                    1.9170e01,
                    -1.1452e00,
                    9.1900e-01,
                    1.1230e00,
                    1.9310e00,
                    4.7790e-02,
                    6.7684e-02,
                    -1.7537e00,
                ],
                [
                    -6.5878e01,
                    1.3500e01,
                    -2.2528e-01,
                    1.8200e00,
                    3.8520e00,
                    1.5450e00,
                    -2.8757e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    -5.4490e00,
                    2.8363e01,
                    -7.7275e-01,
                    2.2360e00,
                    3.7540e00,
                    1.5590e00,
                    -4.6520e00,
                    -7.9736e-03,
                    7.7207e-03,
                ],
            ],
            dtype=torch.float32,
        ),
        box_dim=9,
    )

    gt_labels_3d = np.array([0, 8, 0, 0, 0, 0, -1, 7, 0, 0])
    results = dict(
        pts_filename="tests/data/kitti/a.bin",
        ann_info=dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d),
        bbox3d_fields=[],
        img_fields=[],
    )

    output = pipeline(results)

    expected_tensor = torch.tensor(
        [
            [
                -3.7849e00,
                -4.1057e01,
                -4.8668e-01,
                2.1064e00,
                4.5039e00,
                1.5813e00,
                -1.6919e00,
                1.0469e-02,
                -4.5533e-02,
            ],
            [
                -2.7010e01,
                -6.7551e00,
                -1.3334e00,
                3.5038e-01,
                4.6786e-01,
                7.9883e-01,
                1.4477e00,
                -5.1440e-04,
                1.8758e-03,
            ],
            [
                -4.5448e00,
                -3.6372e01,
                -6.7942e-01,
                2.4476e00,
                4.0544e00,
                1.7693e00,
                1.4721e00,
                0.0000e00,
                -0.0000e00,
            ],
            [
                -3.1916e01,
                -2.3379e00,
                -1.0788e00,
                1.9858e00,
                3.9400e00,
                1.7601e00,
                -3.6564e-01,
                -3.1333e-02,
                8.1166e-02,
            ],
            [
                -4.5802e01,
                -2.2340e01,
                -2.5213e-02,
                2.0298e00,
                4.5355e00,
                1.9838e00,
                2.8199e00,
                0.0000e00,
                -0.0000e00,
            ],
            [
                -4.5526e00,
                -1.8887e01,
                -1.1114e00,
                2.3730e00,
                4.9554e00,
                1.4005e00,
                -1.5997e00,
                0.0000e00,
                -0.0000e00,
            ],
            [
                -2.5648e01,
                -5.2197e00,
                -1.2987e00,
                1.7928e00,
                2.0318e00,
                2.2678e00,
                1.3100e00,
                -3.8428e-02,
                1.5485e-01,
            ],
            [
                -1.5578e00,
                -1.9657e01,
                -1.1699e00,
                9.3878e-01,
                1.1472e00,
                1.9726e00,
                3.0555e00,
                4.5907e-04,
                1.7928e00,
            ],
            [
                -4.4522e00,
                -2.9166e01,
                -7.8938e-01,
                2.2841e00,
                3.8348e00,
                1.5925e00,
                1.4721e00,
                -7.8371e-03,
                -8.1931e-03,
            ],
        ]
    )
    assert torch.allclose(
        output["gt_bboxes_3d"]._data.tensor, expected_tensor, atol=1e-3
    )
