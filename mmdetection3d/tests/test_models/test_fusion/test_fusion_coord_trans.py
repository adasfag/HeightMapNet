# Copyright (c) OpenMMLab. All rights reserved.
"""Tests coords transformation in fusion modules.

CommandLine:
    pytest tests/test_models/test_fusion/test_fusion_coord_trans.py
"""

import torch

from mmdet3d.models.fusion_layers import apply_3d_transformation


def test_coords_transformation():
    """Test the transformation of 3d coords."""

    # H+R+S+T, not reverse, depth
    img_meta = {
        "pcd_scale_factor": 1.2311e00,
        "pcd_rotation": [
            [8.660254e-01, 0.5, 0],
            [-0.5, 8.660254e-01, 0],
            [0, 0, 1.0e00],
        ],
        "pcd_trans": [1.111e-02, -8.88e-03, 0.0],
        "pcd_horizontal_flip": True,
        "transformation_3d_flow": ["HF", "R", "S", "T"],
    }

    pcd = torch.tensor(
        [
            [-5.2422e00, -2.9757e-01, 4.0021e01],
            [-9.1435e-01, 2.6675e01, -5.5950e00],
            [2.0089e-01, 5.8098e00, -3.5409e01],
            [-1.9461e-01, 3.1309e01, -1.0901e00],
        ]
    )

    pcd_transformed = apply_3d_transformation(pcd, "DEPTH", img_meta, reverse=False)

    expected_tensor = torch.tensor(
        [
            [5.78332345e00, 2.900697e00, 4.92698531e01],
            [-1.5433839e01, 2.8993850e01, -6.8880045e00],
            [-3.77929405e00, 6.061661e00, -4.35920199e01],
            [-1.9053658e01, 3.3491436e01, -1.34202211e00],
        ]
    )

    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)

    # H+R+S+T, reverse, depth
    img_meta = {
        "pcd_scale_factor": 7.07106781e-01,
        "pcd_rotation": [
            [7.07106781e-01, 7.07106781e-01, 0.0],
            [-7.07106781e-01, 7.07106781e-01, 0.0],
            [0.0, 0.0, 1.0e00],
        ],
        "pcd_trans": [0.0, 0.0, 0.0],
        "pcd_horizontal_flip": False,
        "transformation_3d_flow": ["HF", "R", "S", "T"],
    }

    pcd = torch.tensor(
        [
            [-5.2422e00, -2.9757e-01, 4.0021e01],
            [-9.1435e01, 2.6675e01, -5.5950e00],
            [6.061661e00, -0.0, -1.0e02],
        ]
    )

    pcd_transformed = apply_3d_transformation(pcd, "DEPTH", img_meta, reverse=True)

    expected_tensor = torch.tensor(
        [
            [-5.53977e00, 4.94463e00, 5.65982409e01],
            [-6.476e01, 1.1811e02, -7.91252488e00],
            [6.061661e00, -6.061661e00, -1.41421356e02],
        ]
    )
    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)

    # H+R+S+T, not reverse, camera
    img_meta = {
        "pcd_scale_factor": 1.0 / 7.07106781e-01,
        "pcd_rotation": [
            [7.07106781e-01, 0.0, 7.07106781e-01],
            [0.0, 1.0e00, 0.0],
            [-7.07106781e-01, 0.0, 7.07106781e-01],
        ],
        "pcd_trans": [1.0e00, -1.0e00, 0.0],
        "pcd_horizontal_flip": True,
        "transformation_3d_flow": ["HF", "S", "R", "T"],
    }

    pcd = torch.tensor(
        [
            [-5.2422e00, 4.0021e01, -2.9757e-01],
            [-9.1435e01, -5.5950e00, 2.6675e01],
            [6.061661e00, -1.0e02, -0.0],
        ]
    )

    pcd_transformed = apply_3d_transformation(pcd, "CAMERA", img_meta, reverse=False)

    expected_tensor = torch.tensor(
        [
            [6.53977e00, 5.55982409e01, 4.94463e00],
            [6.576e01, -8.91252488e00, 1.1811e02],
            [-5.061661e00, -1.42421356e02, -6.061661e00],
        ]
    )

    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)

    # V, reverse, camera
    img_meta = {"pcd_vertical_flip": True, "transformation_3d_flow": ["VF"]}

    pcd_transformed = apply_3d_transformation(pcd, "CAMERA", img_meta, reverse=True)

    expected_tensor = torch.tensor(
        [
            [-5.2422e00, 4.0021e01, 2.9757e-01],
            [-9.1435e01, -5.5950e00, -2.6675e01],
            [6.061661e00, -1.0e02, 0.0],
        ]
    )

    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)

    # V+H, not reverse, depth
    img_meta = {
        "pcd_vertical_flip": True,
        "pcd_horizontal_flip": True,
        "transformation_3d_flow": ["VF", "HF"],
    }

    pcd_transformed = apply_3d_transformation(pcd, "DEPTH", img_meta, reverse=False)

    expected_tensor = torch.tensor(
        [
            [5.2422e00, -4.0021e01, -2.9757e-01],
            [9.1435e01, 5.5950e00, 2.6675e01],
            [-6.061661e00, 1.0e02, 0.0],
        ]
    )
    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)

    # V+H, reverse, lidar
    img_meta = {
        "pcd_vertical_flip": True,
        "pcd_horizontal_flip": True,
        "transformation_3d_flow": ["VF", "HF"],
    }

    pcd_transformed = apply_3d_transformation(pcd, "LIDAR", img_meta, reverse=True)

    expected_tensor = torch.tensor(
        [
            [5.2422e00, -4.0021e01, -2.9757e-01],
            [9.1435e01, 5.5950e00, 2.6675e01],
            [-6.061661e00, 1.0e02, 0.0],
        ]
    )
    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)
