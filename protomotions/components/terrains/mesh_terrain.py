# SPDX-License-Identifier: Apache-2.0
"""External triangle-mesh terrain with matching raster height queries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

from protomotions.components.terrains.terrain import Terrain
from protomotions.utils import rotations


def rasterize_legacy_vertex_heightfield(
    vertices: np.ndarray,
    *,
    origin_xy: np.ndarray,
    shape: tuple[int, int],
    horizontal_scale: float,
) -> np.ndarray:
    """Reproduce the old topology-dependent raster for legacy checkpoints."""
    from scipy.interpolate import griddata

    vertices = np.asarray(vertices, dtype=np.float64)
    rows, cols = shape
    xs = float(origin_xy[0]) + np.arange(rows) * float(horizontal_scale)
    ys = float(origin_xy[1]) + np.arange(cols) * float(horizontal_scale)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    heights = griddata(
        vertices[:, :2], vertices[:, 2], (grid_x, grid_y), method="linear"
    )
    missing = ~np.isfinite(heights)
    if missing.any():
        heights[missing] = griddata(
            vertices[:, :2],
            vertices[:, 2],
            (grid_x[missing], grid_y[missing]),
            method="nearest",
        )
    return np.asarray(heights, dtype=np.float32)


def rasterize_top_surface_heightfield(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    origin_xy: np.ndarray,
    shape: tuple[int, int],
    horizontal_scale: float,
    fallback_height: float | None = None,
) -> np.ndarray:
    """Rasterize triangle surfaces into a topmost 2.5-D height field.

    Unlike interpolation from mesh vertices, this evaluates the piecewise
    linear triangle surfaces at every covered grid point.  The result is
    therefore invariant to retriangulating an unchanged surface.  When
    several surfaces overlap in XY, the highest one is retained, matching the
    support-height semantics of the terrain observation.

    Triangles with zero projected XY area (for example exactly vertical
    walls) cannot be represented in a height field and are ignored.  Grid
    points outside every projected triangle receive ``fallback_height``;
    by default the lowest mesh vertex is used.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    origin_xy = np.asarray(origin_xy, dtype=np.float64)
    rows, cols = (int(shape[0]), int(shape[1]))
    scale = float(horizontal_scale)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vertices must be a non-empty (N, 3) array")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (F, 3)")
    if rows < 2 or cols < 2:
        raise ValueError("height-field shape must be at least (2, 2)")
    if scale <= 0.0:
        raise ValueError("horizontal_scale must be positive")
    if origin_xy.shape != (2,):
        raise ValueError("origin_xy must have shape (2,)")
    if len(triangles) and (
        int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices)
    ):
        raise ValueError("triangle index is outside the vertex array")

    heights = np.full((rows, cols), -np.inf, dtype=np.float64)
    inside_tolerance = 1.0e-9
    projected_area_tolerance = max(scale * scale * 1.0e-12, 1.0e-15)

    for indices in triangles:
        triangle = vertices[indices]
        a, b, c = triangle
        edge_ab = b[:2] - a[:2]
        edge_ac = c[:2] - a[:2]
        determinant = edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0]
        if abs(determinant) <= projected_area_tolerance:
            continue

        lower = np.min(triangle[:, :2], axis=0)
        upper = np.max(triangle[:, :2], axis=0)
        row_start = max(
            0,
            int(np.ceil((lower[0] - origin_xy[0]) / scale - inside_tolerance)),
        )
        row_stop = min(
            rows - 1,
            int(np.floor((upper[0] - origin_xy[0]) / scale + inside_tolerance)),
        )
        col_start = max(
            0,
            int(np.ceil((lower[1] - origin_xy[1]) / scale - inside_tolerance)),
        )
        col_stop = min(
            cols - 1,
            int(np.floor((upper[1] - origin_xy[1]) / scale + inside_tolerance)),
        )
        if row_start > row_stop or col_start > col_stop:
            continue

        row_indices = np.arange(row_start, row_stop + 1)
        col_indices = np.arange(col_start, col_stop + 1)
        grid_rows, grid_cols = np.meshgrid(
            row_indices, col_indices, indexing="ij"
        )
        point_x = origin_xy[0] + grid_rows * scale
        point_y = origin_xy[1] + grid_cols * scale
        offset_x = point_x - a[0]
        offset_y = point_y - a[1]

        weight_b = (offset_x * edge_ac[1] - offset_y * edge_ac[0]) / determinant
        weight_c = (edge_ab[0] * offset_y - edge_ab[1] * offset_x) / determinant
        weight_a = 1.0 - weight_b - weight_c
        inside = (
            (weight_a >= -inside_tolerance)
            & (weight_b >= -inside_tolerance)
            & (weight_c >= -inside_tolerance)
        )
        if not inside.any():
            continue

        triangle_heights = (
            weight_a * a[2] + weight_b * b[2] + weight_c * c[2]
        )
        target = heights[
            row_start : row_stop + 1,
            col_start : col_stop + 1,
        ]
        np.maximum(target, np.where(inside, triangle_heights, -np.inf), out=target)

    missing = ~np.isfinite(heights)
    if missing.any():
        fill = float(np.min(vertices[:, 2])) if fallback_height is None else float(fallback_height)
        heights[missing] = fill
    return heights.astype(np.float32)


def rasterize_surface_height_layers(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    origin_xy: np.ndarray,
    shape: tuple[int, int],
    horizontal_scale: float,
    vertical_merge_tolerance: float = 1.0e-4,
) -> np.ndarray:
    """Rasterize every non-vertical triangle layer at each XY grid point.

    The returned array has shape ``(rows, cols, K)``. Finite values are sorted
    from low to high; unused entries are NaN. Coplanar triangle-edge samples
    are deduplicated so K reflects geometric layers rather than tessellation.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    origin_xy = np.asarray(origin_xy, dtype=np.float64)
    rows, cols = (int(shape[0]), int(shape[1]))
    scale = float(horizontal_scale)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vertices must be a non-empty (N, 3) array")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (F, 3)")
    if rows < 2 or cols < 2:
        raise ValueError("height-field shape must be at least (2, 2)")
    if scale <= 0.0:
        raise ValueError("horizontal_scale must be positive")
    if origin_xy.shape != (2,):
        raise ValueError("origin_xy must have shape (2,)")
    if len(triangles) and (
        int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices)
    ):
        raise ValueError("triangle index is outside the vertex array")

    cell_values: list[list[list[float]]] = [
        [[] for _ in range(cols)] for _ in range(rows)
    ]
    inside_tolerance = 1.0e-9
    projected_area_tolerance = max(scale * scale * 1.0e-12, 1.0e-15)
    for indices in triangles:
        triangle = vertices[indices]
        a, b, c = triangle
        edge_ab = b[:2] - a[:2]
        edge_ac = c[:2] - a[:2]
        determinant = edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0]
        if abs(determinant) <= projected_area_tolerance:
            continue

        lower = np.min(triangle[:, :2], axis=0)
        upper = np.max(triangle[:, :2], axis=0)
        row_start = max(
            0,
            int(np.ceil((lower[0] - origin_xy[0]) / scale - inside_tolerance)),
        )
        row_stop = min(
            rows - 1,
            int(np.floor((upper[0] - origin_xy[0]) / scale + inside_tolerance)),
        )
        col_start = max(
            0,
            int(np.ceil((lower[1] - origin_xy[1]) / scale - inside_tolerance)),
        )
        col_stop = min(
            cols - 1,
            int(np.floor((upper[1] - origin_xy[1]) / scale + inside_tolerance)),
        )
        if row_start > row_stop or col_start > col_stop:
            continue

        row_indices = np.arange(row_start, row_stop + 1)
        col_indices = np.arange(col_start, col_stop + 1)
        grid_rows, grid_cols = np.meshgrid(
            row_indices, col_indices, indexing="ij"
        )
        point_x = origin_xy[0] + grid_rows * scale
        point_y = origin_xy[1] + grid_cols * scale
        offset_x = point_x - a[0]
        offset_y = point_y - a[1]
        weight_b = (offset_x * edge_ac[1] - offset_y * edge_ac[0]) / determinant
        weight_c = (edge_ab[0] * offset_y - edge_ab[1] * offset_x) / determinant
        weight_a = 1.0 - weight_b - weight_c
        inside = (
            (weight_a >= -inside_tolerance)
            & (weight_b >= -inside_tolerance)
            & (weight_c >= -inside_tolerance)
        )
        triangle_heights = (
            weight_a * a[2] + weight_b * b[2] + weight_c * c[2]
        )
        for row, col, height in zip(
            grid_rows[inside].tolist(),
            grid_cols[inside].tolist(),
            triangle_heights[inside].tolist(),
        ):
            cell_values[row][col].append(float(height))

    tolerance = max(float(vertical_merge_tolerance), 0.0)
    unique_values: list[list[np.ndarray]] = []
    maximum_layers = 1
    for row in cell_values:
        unique_row: list[np.ndarray] = []
        for values in row:
            ordered = np.sort(np.asarray(values, dtype=np.float64))
            if len(ordered):
                keep = np.ones(len(ordered), dtype=bool)
                keep[1:] = np.diff(ordered) > tolerance
                ordered = ordered[keep]
            unique_row.append(ordered)
            maximum_layers = max(maximum_layers, len(ordered))
        unique_values.append(unique_row)

    layers = np.full((rows, cols, maximum_layers), np.nan, dtype=np.float32)
    for row_index, row in enumerate(unique_values):
        for col_index, values in enumerate(row):
            layers[row_index, col_index, : len(values)] = values
    return layers


def query_height_layers_bilinear_numpy(
    height_layers: np.ndarray,
    points: np.ndarray,
    *,
    origin_xy: np.ndarray,
    horizontal_scale: float,
    support_ceiling_tolerance: float,
    fallback_height: float,
) -> np.ndarray:
    """Query the nearest rasterized surface not above each point's ceiling."""
    layers = np.asarray(height_layers, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if layers.ndim != 3 or layers.shape[0] < 2 or layers.shape[1] < 2:
        raise ValueError("height_layers must have shape (rows>=2, cols>=2, K)")
    if points.shape[-1] not in (2, 3):
        raise ValueError("points must end in XY or XYZ")
    output_shape = points.shape[:-1]
    flat = points.reshape(-1, points.shape[-1])
    grid = (flat[:, :2] - np.asarray(origin_xy, dtype=np.float32)) / float(
        horizontal_scale
    )
    gx = np.clip(grid[:, 0], 0.0, layers.shape[0] - 1.0001)
    gy = np.clip(grid[:, 1], 0.0, layers.shape[1] - 1.0001)
    x0 = np.minimum(np.floor(gx).astype(np.int64), layers.shape[0] - 2)
    y0 = np.minimum(np.floor(gy).astype(np.int64), layers.shape[1] - 2)
    corners = np.stack(
        (
            layers[x0, y0],
            layers[x0 + 1, y0],
            layers[x0, y0 + 1],
            layers[x0 + 1, y0 + 1],
        ),
        axis=1,
    )
    ceiling = (
        flat[:, 2:3] + float(support_ceiling_tolerance)
        if flat.shape[1] == 3
        else np.full((len(flat), 1), np.inf, dtype=np.float32)
    )
    valid = np.isfinite(corners) & (corners <= ceiling[:, None, :])
    selected = np.max(np.where(valid, corners, -np.inf), axis=2)
    selected[~np.isfinite(selected)] = float(fallback_height)
    fx = gx - x0
    fy = gy - y0
    values = (
        selected[:, 0] * (1.0 - fx) * (1.0 - fy)
        + selected[:, 1] * fx * (1.0 - fy)
        + selected[:, 2] * (1.0 - fx) * fy
        + selected[:, 3] * fx * fy
    )
    return values.reshape(output_shape).astype(np.float32)


class MeshTerrain(Terrain):
    """Use a tiled external mesh for collision and canonical height observations."""

    def __init__(self, config, num_envs: int, device) -> None:
        mesh_path = Path(config.mesh_path).expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Terrain mesh does not exist: {mesh_path}")

        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            if not mesh.geometry:
                raise ValueError(f"Terrain mesh scene is empty: {mesh_path}")
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"Expected a triangle mesh with faces: {mesh_path}")

        self.config = config
        self.device = device
        self.sim_config = config.sim_config
        self.horizontal_scale = float(config.horizontal_scale)
        self.vertical_scale = float(config.vertical_scale)
        self.spacing_between_scenes = config.spacing_between_scenes
        self.minimal_humanoid_spacing = config.minimal_humanoid_spacing
        self.num_scene_slots = 0
        self.num_scenes_per_column = 1

        self.vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        self.triangles = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
        self.origin_xy = bounds[0, :2].copy()

        span = bounds[1, :2] - bounds[0, :2]
        # A single world-space mesh combined with preserve_reference_world_position
        # used to place every parallel humanoid at exactly the same XY location.
        # Isaac Gym collision groups reject cross-environment contacts only after
        # broadphase, so this produced O(num_envs^2) candidate pairs.  Keep the
        # canonical mesh/query grid once, but create several spatial collision
        # copies and assign environments to them in a balanced way.
        requested_tiles = int(getattr(config, "mesh_collision_tiles", 0))
        if requested_tiles <= 0:
            requested_tiles = int(np.ceil(max(1, int(num_envs)) / 16))
        requested_tiles = max(1, requested_tiles)
        requested_tiles = min(requested_tiles, max(1, int(num_envs)))
        tile_cols = int(np.ceil(np.sqrt(requested_tiles)))
        tile_rows = int(np.ceil(requested_tiles / tile_cols))
        # Use a complete rectangle.  The few extra copies for a non-square request
        # keep world-to-canonical mapping branch-free and deterministic.
        self.num_collision_tiles = tile_rows * tile_cols
        margin = max(
            0.0, float(getattr(config, "mesh_collision_tile_margin", 1.0))
        )
        self.collision_tile_step_xy = np.maximum(span + margin, 1.0).astype(
            np.float32
        )
        self.collision_tile_offsets = np.asarray(
            [
                [
                    col * self.collision_tile_step_xy[0],
                    row * self.collision_tile_step_xy[1],
                    0.0,
                ]
                for row in range(tile_rows)
                for col in range(tile_cols)
            ],
            dtype=np.float32,
        )
        self._tile_rows = tile_rows
        self._tile_cols = tile_cols
        tile_ids = torch.arange(num_envs, device=device) % self.num_collision_tiles
        collision_tile_offsets_t = torch.as_tensor(
            self.collision_tile_offsets, device=device, dtype=torch.float32
        )
        self.env_tile_offsets = collision_tile_offsets_t[tile_ids]
        self._canonical_center_xy = torch.as_tensor(
            (bounds[0, :2] + bounds[1, :2]) * 0.5,
            device=device,
            dtype=torch.float32,
        )
        self._collision_tile_step_xy_t = torch.as_tensor(
            self.collision_tile_step_xy, device=device, dtype=torch.float32
        )
        rows = max(2, int(np.ceil(span[0] / self.horizontal_scale)) + 1)
        cols = max(2, int(np.ceil(span[1] / self.horizontal_scale)) + 1)
        # Frozen configs written before this field existed must keep the old
        # observation to remain compatible with their trained normalizer and
        # critic. Newly constructed TerrainConfig objects use the corrected
        # triangle-surface rasterizer by default.
        # Pickled dataclass instances created before the field was introduced
        # inherit the new class attribute when loaded, so inspect their stored
        # instance state instead of plain getattr for reliable compatibility.
        config_state = vars(config) if hasattr(config, "__dict__") else {}
        rasterizer = config_state.get(
            "mesh_height_rasterizer",
            getattr(config, "mesh_height_rasterizer", "legacy_vertex_griddata")
            if not config_state
            else "legacy_vertex_griddata",
        )
        self.fallback_height = float(bounds[0, 2])
        self.support_ceiling_tolerance = float(
            config_state.get(
                "mesh_support_ceiling_tolerance",
                getattr(config, "mesh_support_ceiling_tolerance", 0.05),
            )
        )
        self._layered_support_query = rasterizer == "triangle_surface_layers_v1"
        if self._layered_support_query:
            height_layers = rasterize_surface_height_layers(
                self.vertices,
                self.triangles,
                origin_xy=self.origin_xy,
                shape=(rows, cols),
                horizontal_scale=self.horizontal_scale,
            )
            finite_layers = np.where(
                np.isfinite(height_layers), height_layers, -np.inf
            )
            heights = np.max(finite_layers, axis=2)
            heights[~np.isfinite(heights)] = self.fallback_height
        elif rasterizer == "triangle_surface_v1":
            heights = rasterize_top_surface_heightfield(
                self.vertices,
                self.triangles,
                origin_xy=self.origin_xy,
                shape=(rows, cols),
                horizontal_scale=self.horizontal_scale,
                fallback_height=float(bounds[0, 2]),
            )
            height_layers = heights[..., None]
        elif rasterizer == "legacy_vertex_griddata":
            heights = rasterize_legacy_vertex_heightfield(
                self.vertices,
                origin_xy=self.origin_xy,
                shape=(rows, cols),
                horizontal_scale=self.horizontal_scale,
            )
            height_layers = heights[..., None]
        else:
            raise ValueError(
                "Unsupported mesh_height_rasterizer="
                f"{rasterizer!r}; expected 'triangle_surface_layers_v1', "
                "'triangle_surface_v1', or "
                "'legacy_vertex_griddata'"
            )
        self.mesh_height_rasterizer = rasterizer

        self.height_layers = torch.as_tensor(
            height_layers, device=device, dtype=torch.float32
        )
        self.height_samples = torch.as_tensor(
            heights, device=device, dtype=torch.float32
        )
        self.height_field_raw = np.rint(heights / self.vertical_scale).astype(
            np.int16
        )
        self.tot_rows, self.tot_cols = self.height_field_raw.shape
        self.border = 0
        self.border_size = 0.0
        self.object_playground_cols = 0
        self.object_playground_buffer_size = 0
        self.scene_y_offset = 0.0
        self.scene_placement_map = torch.zeros(
            self.tot_rows, self.tot_cols, dtype=torch.bool, device=device
        )

        self.walkable_x_coords = torch.as_tensor(
            self.vertices[:, 0], dtype=torch.float32, device=device
        )
        self.walkable_y_coords = torch.as_tensor(
            self.vertices[:, 1], dtype=torch.float32, device=device
        )
        self.flat_x_coords = self.walkable_x_coords
        self.flat_y_coords = self.walkable_y_coords
        self.num_height_points, self.height_points = self.init_height_points(num_envs)

        print(
            f"Loaded mesh terrain {mesh_path}: "
            f"{len(self.vertices)} vertices, {len(self.triangles)} triangles, "
            f"height grid={rows}x{cols}, rasterizer={rasterizer}, "
            f"origin_xy={self.origin_xy.tolist()}, "
            f"collision_tiles={self.num_collision_tiles} "
            f"({self._tile_rows}x{self._tile_cols})"
        )

    def is_flat(self) -> bool:
        return False

    def sample_valid_locations(self, num_envs, sample_flat=False):
        indices = torch.randint(
            0, len(self.walkable_x_coords), (num_envs,), device=self.device
        )
        locations = torch.stack(
            [self.walkable_x_coords[indices], self.walkable_y_coords[indices]], dim=-1
        )
        # Preserve the original random-spawn behavior for non-interaction tasks,
        # while spreading those samples over the replicated collision scenes.
        tile_ids = torch.randint(
            0, self.num_collision_tiles, (num_envs,), device=self.device
        )
        tile_offsets = torch.as_tensor(
            self.collision_tile_offsets, device=self.device, dtype=locations.dtype
        )
        return locations + tile_offsets[tile_ids, :2]

    def sample_flat_locations(self, num_envs):
        return self.sample_valid_locations(num_envs, sample_flat=False)

    def is_valid_spawn_location(self, locations: torch.Tensor) -> torch.Tensor:
        locations = self._world_to_canonical(locations)
        lower = torch.as_tensor(self.origin_xy, device=locations.device)
        upper = lower + torch.tensor(
            [
                (self.tot_rows - 1) * self.horizontal_scale,
                (self.tot_cols - 1) * self.horizontal_scale,
            ],
            device=locations.device,
        )
        return ((locations >= lower) & (locations <= upper)).all(dim=-1)

    def mark_scene_location(self, x, y):
        return None

    def get_env_offsets(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the balanced collision-scene translation for each environment."""
        return self.env_tile_offsets[env_ids]

    def _world_to_canonical(self, locations: torch.Tensor) -> torch.Tensor:
        """Map points near any collision tile back to the canonical mesh frame."""
        if self.num_collision_tiles == 1:
            return locations
        points = locations.clone()
        center = self._canonical_center_xy.to(device=points.device, dtype=points.dtype)
        step = self._collision_tile_step_xy_t.to(device=points.device, dtype=points.dtype)
        tile_xy = torch.round((points[..., :2] - center) / step)
        tile_x = tile_xy[..., 0].clamp(0, self._tile_cols - 1)
        tile_y = tile_xy[..., 1].clamp(0, self._tile_rows - 1)
        points[..., 0] -= tile_x * step[0]
        points[..., 1] -= tile_y * step[1]
        return points

    def _query_heights(self, locations: torch.Tensor) -> torch.Tensor:
        original_shape = locations.shape[:-1]
        points = self._world_to_canonical(locations).reshape(
            -1, locations.shape[-1]
        )
        origin = torch.as_tensor(
            self.origin_xy, device=points.device, dtype=points.dtype
        )
        grid = (points[:, :2] - origin) / self.horizontal_scale
        gx = grid[:, 0].clamp(0, self.height_samples.shape[0] - 1.0001)
        gy = grid[:, 1].clamp(0, self.height_samples.shape[1] - 1.0001)
        x0 = gx.floor().long().clamp(max=self.height_samples.shape[0] - 2)
        y0 = gy.floor().long().clamp(max=self.height_samples.shape[1] - 2)
        fx = gx - x0.float()
        fy = gy - y0.float()
        if self._layered_support_query and points.shape[1] >= 3:
            corners = torch.stack(
                (
                    self.height_layers[x0, y0],
                    self.height_layers[x0 + 1, y0],
                    self.height_layers[x0, y0 + 1],
                    self.height_layers[x0 + 1, y0 + 1],
                ),
                dim=1,
            )
            ceiling = points[:, 2:3] + self.support_ceiling_tolerance
            valid = torch.isfinite(corners) & (corners <= ceiling[:, None, :])
            selected = torch.where(
                valid,
                corners,
                torch.full_like(corners, -torch.inf),
            ).amax(dim=2)
            selected = torch.where(
                torch.isfinite(selected),
                selected,
                torch.full_like(selected, self.fallback_height),
            )
            h00, h10, h01, h11 = selected.unbind(dim=1)
        else:
            h00 = self.height_samples[x0, y0]
            h10 = self.height_samples[x0 + 1, y0]
            h01 = self.height_samples[x0, y0 + 1]
            h11 = self.height_samples[x0 + 1, y0 + 1]
        heights = (
            h00 * (1 - fx) * (1 - fy)
            + h10 * fx * (1 - fy)
            + h01 * (1 - fx) * fy
            + h11 * fx * fy
        )
        return heights.reshape(original_shape)

    def get_ground_heights(self, locations: torch.Tensor) -> torch.Tensor:
        if locations.ndim == 2:
            return self._query_heights(locations).unsqueeze(-1)
        return self._query_heights(locations)

    def get_height_maps(self, root_states, env_ids=None, return_all_dims=False):
        points_local = (
            self.height_points[env_ids].clone()
            if env_ids is not None
            else self.height_points.clone()
        )
        points = rotations.quat_apply_yaw(
            root_states.root_rot.repeat(1, self.num_height_points),
            points_local,
            True,
        ) + root_states.root_pos[:, None, :3]
        heights = self._query_heights(points)
        if return_all_dims:
            return torch.cat([points[..., :2], heights.unsqueeze(-1)], dim=-1)
        return root_states.root_pos[:, 2:3] - heights
