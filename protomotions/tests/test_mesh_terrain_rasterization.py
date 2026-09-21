from __future__ import annotations

import numpy as np
import torch

from protomotions.components.terrains.mesh_terrain import (
    MeshTerrain,
    query_height_layers_bilinear_numpy,
    rasterize_surface_height_layers,
    rasterize_top_surface_heightfield,
)
from protomotions.components.terrains.config import TerrainConfig


def _raster(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return rasterize_top_surface_heightfield(
        vertices,
        faces,
        origin_xy=np.asarray([0.0, 0.0]),
        shape=(11, 11),
        horizontal_scale=0.1,
    )


def test_raster_is_invariant_to_planar_retriangulation() -> None:
    # Both triangulations are exactly the same plane z = 0.3 + 0.2x - 0.1y.
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.3],
            [1.0, 0.0, 0.5],
            [1.0, 1.0, 0.4],
            [0.0, 1.0, 0.2],
        ],
        dtype=np.float32,
    )
    diagonal_ac = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    diagonal_bd = np.asarray([[0, 1, 3], [1, 2, 3]], dtype=np.int64)

    first = _raster(vertices, diagonal_ac)
    second = _raster(vertices, diagonal_bd)
    np.testing.assert_allclose(first, second, atol=1.0e-6)

    x = np.arange(11, dtype=np.float32)[:, None] * 0.1
    y = np.arange(11, dtype=np.float32)[None, :] * 0.1
    np.testing.assert_allclose(first, 0.3 + 0.2 * x - 0.1 * y, atol=1.0e-6)


def test_raster_uses_topmost_overlapping_surface() -> None:
    bottom = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
    )
    top = bottom.copy()
    top[:, 2] = 1.25
    vertices = np.concatenate((bottom, top), axis=0)
    faces = np.asarray(
        [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], dtype=np.int64
    )

    np.testing.assert_allclose(_raster(vertices, faces), 1.25, atol=1.0e-6)


def test_layered_query_selects_nearest_surface_below_query() -> None:
    bottom = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
    )
    top = bottom.copy()
    top[:, 2] = 1.25
    vertices = np.concatenate((bottom, top), axis=0)
    faces = np.asarray(
        [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], dtype=np.int64
    )
    layers = rasterize_surface_height_layers(
        vertices,
        faces,
        origin_xy=np.asarray([0.0, 0.0]),
        shape=(11, 11),
        horizontal_scale=0.1,
    )
    query = np.asarray(
        [[0.5, 0.5, 0.7], [0.5, 0.5, 1.5], [0.5, 0.5, 1.21]],
        dtype=np.float32,
    )
    height = query_height_layers_bilinear_numpy(
        layers,
        query,
        origin_xy=np.asarray([0.0, 0.0]),
        horizontal_scale=0.1,
        support_ceiling_tolerance=0.05,
        fallback_height=-1.0,
    )
    np.testing.assert_allclose(height, [0.0, 1.25, 1.25], atol=1.0e-6)


def test_layered_raster_is_invariant_to_planar_retriangulation() -> None:
    vertices = np.asarray(
        [[0, 0, 0.3], [1, 0, 0.5], [1, 1, 0.4], [0, 1, 0.2]],
        dtype=np.float32,
    )
    first = rasterize_surface_height_layers(
        vertices,
        np.asarray([[0, 1, 2], [0, 2, 3]]),
        origin_xy=np.asarray([0.0, 0.0]),
        shape=(11, 11),
        horizontal_scale=0.1,
    )
    second = rasterize_surface_height_layers(
        vertices,
        np.asarray([[0, 1, 3], [1, 2, 3]]),
        origin_xy=np.asarray([0.0, 0.0]),
        shape=(11, 11),
        horizontal_scale=0.1,
    )
    np.testing.assert_allclose(first, second, atol=1.0e-6, equal_nan=True)


def test_mesh_collision_tiles_preserve_canonical_height_queries(tmp_path) -> None:
    mesh_path = tmp_path / "plane.obj"
    mesh_path.write_text(
        "\n".join(
            (
                "v 0 0 0.2",
                "v 1 0 0.2",
                "v 1 1 0.2",
                "v 0 1 0.2",
                "f 1 2 3",
                "f 1 3 4",
            )
        )
    )
    config = TerrainConfig(
        mesh_path=str(mesh_path),
        horizontal_scale=0.1,
        num_samples_per_axis=2,
        mesh_collision_tiles=4,
        mesh_collision_tile_margin=0.5,
    )
    terrain = MeshTerrain(config, num_envs=8, device=torch.device("cpu"))

    np.testing.assert_allclose(
        terrain.collision_tile_offsets[:, :2],
        [[0.0, 0.0], [1.5, 0.0], [0.0, 1.5], [1.5, 1.5]],
    )
    env_ids = torch.arange(8)
    expected_offsets = torch.as_tensor(terrain.collision_tile_offsets).repeat(2, 1)
    torch.testing.assert_close(terrain.get_env_offsets(env_ids), expected_offsets)

    canonical = torch.tensor([[0.4, 0.6, 1.0]])
    tiled = canonical + torch.as_tensor(terrain.collision_tile_offsets)
    expected_height = torch.full((4,), 0.2)
    torch.testing.assert_close(terrain._query_heights(tiled), expected_height)
