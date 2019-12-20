import math

import imageio
import numpy as np
import torch


def to_gpu(data, device=None):
    if isinstance(data, tuple) or isinstance(data, list):
        return [torch.as_tensor(d).cuda(device) for d in data]
    else:
        return torch.as_tensor(data).cuda(device)


def imread(filename):
    return imageio.imread(filename).astype('float32') / 255.


def create_textures(num_faces, texture_size=16, flatten=False):
    if not flatten:
        tile_width = int((num_faces - 1.) ** 0.5) + 1
        tile_height = int((num_faces - 1.) / tile_width) + 1
    else:
        tile_width = 1
        tile_height = num_faces
    textures = np.ones((3, tile_height * texture_size, tile_width * texture_size), 'float32')

    vertices = np.zeros((num_faces, 3, 2), 'float32')  # [:, :, XY]
    face_nums = np.arange(num_faces)
    column = face_nums % tile_width
    row = face_nums // tile_width
    vertices[:, 0, 0] = column * texture_size
    vertices[:, 0, 1] = row * texture_size
    vertices[:, 1, 0] = column * texture_size
    vertices[:, 1, 1] = (row + 1) * texture_size - 1
    vertices[:, 2, 0] = (column + 1) * texture_size - 1
    vertices[:, 2, 1] = (row + 1) * texture_size - 1
    vertices = vertices.reshape((num_faces * 3, 2))
    faces = np.arange(num_faces * 3).reshape((num_faces, 3)).astype('int32')

    return vertices, faces, textures


def radians(degrees):
    pi = 3.14159265359
    return degrees / 180. * pi


def get_points_from_angles(distance, elevation, azimuth, degrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = math.radians(elevation)
            azimuth = math.radians(azimuth)
        return (
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            -distance * math.cos(elevation) * math.cos(azimuth))
    else:
        if degrees:
            elevation = radians(elevation)
            azimuth = radians(azimuth)
        return torch.stack([
            distance * torch.cos(elevation) * torch.sin(azimuth),
            distance * torch.sin(elevation),
            -distance * torch.cos(elevation) * torch.cos(azimuth),
        ]).permute(1, 0)


def pad_zeros(x, size, axis, side='both'):
    if axis == 1:
        pad = torch.zeros((x.shape[0], size, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device)
    elif axis == 2:
        pad = torch.zeros((x.shape[0], x.shape[1], size, x.shape[3]), dtype=torch.float32, device=x.device)
    elif axis == 3:
        pad = torch.zeros((x.shape[0], x.shape[1], x.shape[2], size), dtype=torch.float32, device=x.device)
    if side == 'both':
        x = torch.cat((pad, x, pad), axis)
    elif side == 'left':
        x = torch.cat((pad, x), axis)
    elif side == 'right':
        x = torch.cat((x, pad), axis)
    return x


def maximum(data_right, data_left, eps=1e-4):
    max_map = torch.max(data_left, data_right) <= 0
    min_map = torch.abs(data_right - data_left) < eps
    rl_map = data_right > data_left
    else_map = ~(max_map | min_map | rl_map)

    data3 = torch.zeros_like(data_right)
    data3[max_map] = 0
    data3[min_map] = 0
    data3[rl_map] = -data_right[rl_map]
    data3[else_map] = data_left[else_map]

    return data3
