# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
from typing import Tuple
import torch


def project_cone_beam_vectorized(
    coords3d: torch.Tensor,
    image_plane_vertices: torch.Tensor,
    light_source: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    batch_size = coords3d.shape[0]
    device = coords3d.device
    image_plane_vertices = image_plane_vertices.to(device=device)
    light_source = light_source.to(device=device)

    plane_indices = torch.arange(batch_size, device=device) % image_plane_vertices.shape[0]
    image_plane_vertices_batch = image_plane_vertices[plane_indices]

    top_left = image_plane_vertices_batch[:, 0:1, :]
    bottom_left = image_plane_vertices_batch[:, 1:2, :]
    bottom_right = image_plane_vertices_batch[:, 2:3, :]
    top_right = image_plane_vertices_batch[:, 3:4, :]

    x_dir = ((top_right - top_left) + (bottom_right - bottom_left)) / 2.0
    x_dir = x_dir / (torch.norm(x_dir, dim=-1, keepdim=True) + 1e-8)
    y_dir = ((bottom_left - top_left) + (bottom_right - top_right)) / 2.0
    y_dir = y_dir / (torch.norm(y_dir, dim=-1, keepdim=True) + 1e-8)
    normal = torch.cross(x_dir, y_dir, dim=-1)
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-8)
    plane_center = (top_left + bottom_right) / 2.0
    width = torch.norm(top_right - top_left, dim=-1, keepdim=True)
    height = torch.norm(bottom_left - top_left, dim=-1, keepdim=True)

    light = light_source.unsqueeze(1)
    ray_dirs = coords3d - light
    ray_dirs = ray_dirs / (torch.norm(ray_dirs, dim=-1, keepdim=True) + 1e-8)
    n_dot_ray = torch.sum(normal * ray_dirs, dim=-1, keepdim=True)
    n_dot_light_to_center = torch.sum(normal * (plane_center - light), dim=-1, keepdim=True)
    t = n_dot_light_to_center / (n_dot_ray + 1e-8)
    intersections = light + t * ray_dirs
    local_vec = intersections - top_left
    local_x = torch.sum(local_vec * x_dir, dim=-1, keepdim=True)
    local_y = torch.sum(local_vec * y_dir, dim=-1, keepdim=True)
    pixel_x = (local_x / (width + 1e-8)) * image_size[0]
    pixel_y = (local_y / (height + 1e-8)) * image_size[1]
    return torch.cat([pixel_x, pixel_y], dim=-1)


def pixel_to_3d_on_image_plane(
    coords2d: torch.Tensor,
    image_plane_vertices: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    top_left = image_plane_vertices[:, 0:1, :]
    bottom_left = image_plane_vertices[:, 1:2, :]
    bottom_right = image_plane_vertices[:, 2:3, :]
    top_right = image_plane_vertices[:, 3:4, :]
    x_dir = ((top_right - top_left) + (bottom_right - bottom_left)) / 2.0
    x_dir = x_dir / (torch.norm(x_dir, dim=-1, keepdim=True) + 1e-8)
    y_dir = ((bottom_left - top_left) + (bottom_right - top_right)) / 2.0
    y_dir = y_dir / (torch.norm(y_dir, dim=-1, keepdim=True) + 1e-8)
    width = torch.norm(top_right - top_left, dim=-1, keepdim=True)
    height = torch.norm(bottom_left - top_left, dim=-1, keepdim=True)
    image_size_w = torch.tensor(image_size[0], dtype=coords2d.dtype, device=coords2d.device)
    image_size_h = torch.tensor(image_size[1], dtype=coords2d.dtype, device=coords2d.device)
    local_x = (coords2d[..., 0:1] / image_size_w) * width
    local_y = (coords2d[..., 1:2] / image_size_h) * height
    return top_left + local_x * x_dir + local_y * y_dir


def triangulate_from_two_views(
    coords2d_view1: torch.Tensor,
    coords2d_view2: torch.Tensor,
    image_plane_vertices1: torch.Tensor,
    image_plane_vertices2: torch.Tensor,
    light_source1: torch.Tensor,
    light_source2: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    points_on_plane1 = pixel_to_3d_on_image_plane(coords2d_view1, image_plane_vertices1, image_size)
    points_on_plane2 = pixel_to_3d_on_image_plane(coords2d_view2, image_plane_vertices2, image_size)
    light1 = light_source1.unsqueeze(1)
    light2 = light_source2.unsqueeze(1)
    ray_dir1 = points_on_plane1 - light1
    ray_dir2 = points_on_plane2 - light2
    ray_dir1 = ray_dir1 / (torch.norm(ray_dir1, dim=-1, keepdim=True) + 1e-8)
    ray_dir2 = ray_dir2 / (torch.norm(ray_dir2, dim=-1, keepdim=True) + 1e-8)
    light_diff = light2 - light1
    dot12 = torch.sum(ray_dir1 * ray_dir2, dim=-1, keepdim=True)
    dot1d = torch.sum(ray_dir1 * light_diff, dim=-1, keepdim=True)
    dot2d = torch.sum(ray_dir2 * light_diff, dim=-1, keepdim=True)
    det = 1.0 - dot12 ** 2
    t1 = (dot1d - dot12 * dot2d) / (det + 1e-8)
    t2 = (dot12 * dot1d - dot2d) / (det + 1e-8)
    point1 = light1 + t1 * ray_dir1
    point2 = light2 + t2 * ray_dir2
    return (point1 + point2) / 2.0
