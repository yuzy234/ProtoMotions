# SPDX-License-Identifier: Apache-2.0

import torch

from data.scripts.convert_easymimic_smpl_to_motionlib import infer_surface_contacts


class FlatTerrain:
    def get_ground_heights(self, points):
        return torch.zeros(points.shape[:-1], dtype=points.dtype)


def test_surface_contacts_separate_support_impact_and_fast_near_pass():
    vertices = torch.zeros(5, 6, 3)
    body_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    # Body 0 remains close and slow: support.
    vertices[:, :2, 2] = 0.04
    # Body 1 passes close to the surface too quickly: not support.
    vertices[:, 2:4, 2] = 0.04
    vertices[:, 2:4, 0] = torch.arange(5).float()[:, None]
    # Body 2 is fast but reaches the impact band in the middle frame.
    vertices[:, 4:, 2] = 0.20
    vertices[2, 4:, 2] = 0.01

    contacts, clearance, speed = infer_surface_contacts(
        vertices,
        body_ids,
        FlatTerrain(),
        fps=10.0,
        num_bodies=3,
        contact_distance=0.08,
        penetration_tolerance=0.08,
        speed_threshold=1.0,
        impact_distance=0.02,
        robust_vertex_count=2,
    )

    assert contacts[:, 0].all()
    assert not contacts[:, 1].any()
    assert contacts[2, 2]
    assert contacts[:, 2].sum().item() == 1
    torch.testing.assert_close(clearance[:, 0], torch.full((5,), 0.04))
    assert speed[:, 1].min().item() > 1.0
