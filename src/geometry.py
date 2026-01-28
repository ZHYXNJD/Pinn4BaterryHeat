from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BoxGeometry:
    length_x: float
    length_y: float
    length_z: float

    def normalized(self, coords: torch.Tensor) -> torch.Tensor:
        half = torch.tensor(
            [self.length_x / 2, self.length_y / 2, self.length_z / 2],
            device=coords.device,
            dtype=coords.dtype,
        )
        return coords / half

    def sample_interior(self, n: int, device: torch.device) -> torch.Tensor:
        rand = torch.rand((n, 3), device=device)
        scale = torch.tensor(
            [self.length_x, self.length_y, self.length_z],
            device=device,
            dtype=rand.dtype,
        )
        coords = (rand - 0.5) * scale
        return coords

    def sample_face(self, n: int, device: torch.device, face: int):
        coords = torch.zeros((n, 3), device=device)
        normals = torch.zeros((n, 3), device=device)
        rand = torch.rand((n, 2), device=device)
        half = torch.tensor(
            [self.length_x / 2, self.length_y / 2, self.length_z / 2],
            device=device,
        )
        i = face
        if i == 0:
            coords[:, 0] = -half[0]
            coords[:, 1] = (rand[:, 0] - 0.5) * 2 * half[1]
            coords[:, 2] = (rand[:, 1] - 0.5) * 2 * half[2]
            normals[:, 0] = -1.0
        elif i == 1:
            coords[:, 0] = half[0]
            coords[:, 1] = (rand[:, 0] - 0.5) * 2 * half[1]
            coords[:, 2] = (rand[:, 1] - 0.5) * 2 * half[2]
            normals[:, 0] = 1.0
        elif i == 2:
            coords[:, 1] = -half[1]
            coords[:, 0] = (rand[:, 0] - 0.5) * 2 * half[0]
            coords[:, 2] = (rand[:, 1] - 0.5) * 2 * half[2]
            normals[:, 1] = -1.0
        elif i == 3:
            coords[:, 1] = half[1]
            coords[:, 0] = (rand[:, 0] - 0.5) * 2 * half[0]
            coords[:, 2] = (rand[:, 1] - 0.5) * 2 * half[2]
            normals[:, 1] = 1.0
        elif i == 4:
            coords[:, 2] = -half[2]
            coords[:, 0] = (rand[:, 0] - 0.5) * 2 * half[0]
            coords[:, 1] = (rand[:, 1] - 0.5) * 2 * half[1]
            normals[:, 2] = -1.0
        else:
            coords[:, 2] = half[2]
            coords[:, 0] = (rand[:, 0] - 0.5) * 2 * half[0]
            coords[:, 1] = (rand[:, 1] - 0.5) * 2 * half[1]
            normals[:, 2] = 1.0
        return coords, normals

    def sample_boundary(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        faces = torch.randint(0, 6, (n,), device=device)
        rand = torch.rand((n, 2), device=device)
        coords = torch.zeros((n, 3), device=device)
        normals = torch.zeros((n, 3), device=device)

        half = torch.tensor(
            [self.length_x / 2, self.length_y / 2, self.length_z / 2],
            device=device,
        )

        for i in range(6):
            mask = faces == i
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if i == 0:
                coords[idx, 0] = -half[0]
                coords[idx, 1] = (rand[idx, 0] - 0.5) * 2 * half[1]
                coords[idx, 2] = (rand[idx, 1] - 0.5) * 2 * half[2]
                normals[idx, 0] = -1.0
            elif i == 1:
                coords[idx, 0] = half[0]
                coords[idx, 1] = (rand[idx, 0] - 0.5) * 2 * half[1]
                coords[idx, 2] = (rand[idx, 1] - 0.5) * 2 * half[2]
                normals[idx, 0] = 1.0
            elif i == 2:
                coords[idx, 1] = -half[1]
                coords[idx, 0] = (rand[idx, 0] - 0.5) * 2 * half[0]
                coords[idx, 2] = (rand[idx, 1] - 0.5) * 2 * half[2]
                normals[idx, 1] = -1.0
            elif i == 3:
                coords[idx, 1] = half[1]
                coords[idx, 0] = (rand[idx, 0] - 0.5) * 2 * half[0]
                coords[idx, 2] = (rand[idx, 1] - 0.5) * 2 * half[2]
                normals[idx, 1] = 1.0
            elif i == 4:
                coords[idx, 2] = -half[2]
                coords[idx, 0] = (rand[idx, 0] - 0.5) * 2 * half[0]
                coords[idx, 1] = (rand[idx, 1] - 0.5) * 2 * half[1]
                normals[idx, 2] = -1.0
            else:
                coords[idx, 2] = half[2]
                coords[idx, 0] = (rand[idx, 0] - 0.5) * 2 * half[0]
                coords[idx, 1] = (rand[idx, 1] - 0.5) * 2 * half[1]
                normals[idx, 2] = 1.0
        return coords, normals, faces

    def _half_lengths(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.length_x / 2, self.length_y / 2, self.length_z / 2],
            device=device,
        )

    def _tangent_axes(self, face: int):
        if face in (0, 1):  # x faces
            return (1, 2)
        if face in (2, 3):  # y faces
            return (0, 2)
        return (0, 1)  # z faces

    def sample_layer(
        self,
        n: int,
        device: torch.device,
        face: int,
        thickness: float,
        offset: float = 0.0,
    ) -> torch.Tensor:
        rand = torch.rand((n, 3), device=device)
        half = torch.tensor(
            [self.length_x / 2, self.length_y / 2, self.length_z / 2],
            device=device,
            dtype=rand.dtype,
        )
        coords = torch.zeros((n, 3), device=device)
        tang = (rand[:, :2] - 0.5) * 2
        depth = offset + rand[:, 2] * thickness
        if face == 0:  # x_min
            coords[:, 0] = -half[0] - depth
            coords[:, 1] = tang[:, 0] * half[1]
            coords[:, 2] = tang[:, 1] * half[2]
        elif face == 1:  # x_max
            coords[:, 0] = half[0] + depth
            coords[:, 1] = tang[:, 0] * half[1]
            coords[:, 2] = tang[:, 1] * half[2]
        elif face == 2:  # y_min
            coords[:, 1] = -half[1] - depth
            coords[:, 0] = tang[:, 0] * half[0]
            coords[:, 2] = tang[:, 1] * half[2]
        elif face == 3:  # y_max
            coords[:, 1] = half[1] + depth
            coords[:, 0] = tang[:, 0] * half[0]
            coords[:, 2] = tang[:, 1] * half[2]
        elif face == 4:  # z_min
            coords[:, 2] = -half[2] - depth
            coords[:, 0] = tang[:, 0] * half[0]
            coords[:, 1] = tang[:, 1] * half[1]
        else:  # z_max
            coords[:, 2] = half[2] + depth
            coords[:, 0] = tang[:, 0] * half[0]
            coords[:, 1] = tang[:, 1] * half[1]
        return coords

    def channel_centers(self, axis: int, count: int, pitch: float, device: torch.device) -> torch.Tensor:
        if count <= 0:
            return torch.empty((0,), device=device)
        span = pitch * (count - 1)
        start = -0.5 * span
        centers = start + pitch * torch.arange(count, device=device)
        return centers

    def sample_fluid_layer(
        self,
        n: int,
        device: torch.device,
        face: int,
        thickness: float,
        offset: float = 0.0,
    ) -> torch.Tensor:
        return self.sample_layer(n, device, face, thickness, offset)

    def sample_fluid_inlet(
        self,
        n: int,
        device: torch.device,
        face: int,
        thickness: float,
        flow_dir: int,
        offset: float = 0.0,
    ):
        coords = self.sample_layer(n, device, face, thickness, offset)
        normal_axis = face // 2
        if flow_dir != normal_axis:
            half = self._half_lengths(device)
            coords[:, flow_dir] = -half[flow_dir]
        return coords

    def sample_channel_layer(
        self,
        n: int,
        device: torch.device,
        face: int,
        thickness: float,
        channel_axis: int,
        channel_centers: torch.Tensor,
        channel_width: float,
        offset: float = 0.0,
    ) -> torch.Tensor:
        if channel_centers.numel() == 0 or channel_width <= 0:
            return self.sample_fluid_layer(n, device, face, thickness, offset)

        rand = torch.rand((n, 3), device=device)
        coords = torch.zeros((n, 3), device=device)
        half = self._half_lengths(device)
        tang_axes = self._tangent_axes(face)

        if channel_axis not in tang_axes:
            channel_axis = tang_axes[0]
        other_axis = tang_axes[1] if tang_axes[0] == channel_axis else tang_axes[0]

        center_idx = torch.randint(0, channel_centers.numel(), (n,), device=device)
        centers = channel_centers[center_idx]

        coords[:, channel_axis] = torch.clamp(
            centers + (rand[:, 0] - 0.5) * channel_width, -half[channel_axis], half[channel_axis]
        )
        coords[:, other_axis] = (rand[:, 1] - 0.5) * 2 * half[other_axis]

        depth = offset + rand[:, 2] * thickness
        if face == 0:
            coords[:, 0] = -half[0] - depth
        elif face == 1:
            coords[:, 0] = half[0] + depth
        elif face == 2:
            coords[:, 1] = -half[1] - depth
        elif face == 3:
            coords[:, 1] = half[1] + depth
        elif face == 4:
            coords[:, 2] = -half[2] - depth
        else:
            coords[:, 2] = half[2] + depth
        return coords

    def sample_channel_inlet(
        self,
        n: int,
        device: torch.device,
        face: int,
        thickness: float,
        flow_dir: int,
        channel_axis: int,
        channel_centers: torch.Tensor,
        channel_width: float,
        offset: float = 0.0,
    ) -> torch.Tensor:
        if channel_centers.numel() == 0 or channel_width <= 0:
            return self.sample_fluid_inlet(n, device, face, thickness, flow_dir, offset)

        coords = torch.zeros((n, 3), device=device)
        rand = torch.rand((n, 2), device=device)
        half = self._half_lengths(device)
        tang_axes = self._tangent_axes(face)

        if channel_axis not in tang_axes:
            channel_axis = tang_axes[0]
        other_axis = tang_axes[1] if tang_axes[0] == channel_axis else tang_axes[0]

        center_idx = torch.randint(0, channel_centers.numel(), (n,), device=device)
        centers = channel_centers[center_idx]

        coords[:, channel_axis] = torch.clamp(
            centers + (rand[:, 0] - 0.5) * channel_width, -half[channel_axis], half[channel_axis]
        )
        coords[:, other_axis] = (rand[:, 1] - 0.5) * 2 * half[other_axis]

        normal_axis = face // 2
        if flow_dir != normal_axis:
            coords[:, flow_dir] = -half[flow_dir]
        return coords
