import glob
import os
import subprocess

import imageio
import numpy as np
import torch


def make_gif(working_directory, filename):
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/_tmp_*.png %s' %
                    (options, working_directory, filename), shell=True)
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)


def to_gpu(data, device=None):
    if isinstance(data, tuple) or isinstance(data, list):
        return [torch.as_tensor(d).cuda(device) for d in data]
    else:
        return torch.as_tensor(data).cuda(device)


def imread(filename):
    image = np.asarray(imageio.imread(filename), dtype='float32') / 255.
    return image


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


def get_points_from_angles(distance, elevation, azimuth, degrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = np.radians(elevation)
            azimuth = np.radians(azimuth)
        return (
            distance * np.cos(elevation) * np.sin(azimuth),
            distance * np.sin(elevation),
            -distance * np.cos(elevation) * np.cos(azimuth))
    else:
        if degrees:
            elevation = elevation / 180. * 3.14159265359
            azimuth = azimuth / 180. * 3.14159265359
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
    max_map = torch.max(data_right, data_left) <= 0
    abs_map = torch.abs(data_right - data_left) < eps
    rl_map = data_right > data_left

    data3 = data_left.clone()
    data3[rl_map] = -data_right[rl_map]
    data3[abs_map] = 0
    data3[max_map] = 0

    return data3


def to_map(data_in, indices):
    data_in, indices = data_in.contiguous(), indices.contiguous()
    data_out = torch.zeros((*indices.shape[:3], *data_in.shape[2:]), dtype=torch.float32)
    data_out = data_out.contiguous().to(data_in.device)

    indices_mask = indices >= 0

    for b in range(data_in.shape[0]):
        data_out[b, indices_mask[b]] = data_in[b, indices[b, indices_mask[b]].cpu().numpy(), :]

    return data_out


class MaskForeground(torch.autograd.Function):
    """
    Test code:
    import chainer.gradient_check
    data_in = cp.random.randn(*(16, 3, 3, 5)).astype('float32')
    masks = cp.random.randint(0, 2, size=(16, 3, 3)).astype('int32')
    grad_out = cp.random.randn(16, 3, 3, 5).astype('float32')
    data_out = mask_foreground(data_in, masks)
    for i1 in range(16):
        for i2 in range(3):
            for i3 in range(3):
                i4 = masks[i1, i2, i3]
                if 0 <= i4:
                    chainer.testing.assert_allclose(data_out[i1, i2, i3].data, data_in[i1, i2, i3])
                else:
                    chainer.testing.assert_allclose(data_out[i1, i2, i3].data, data_out[i1, i2, i3].data * 0)
    chainer.gradient_check.check_backward(
        mask_foreground, (data_in, masks), grad_out, no_grads=(False, True), rtol=1e-2, atol=1e-03)
    """

    @staticmethod
    def forward(ctx, data_in, face_index_map):
        data_in, face_index_map = data_in.contiguous(), face_index_map.contiguous()
        data_out = torch.zeros(data_in.shape, dtype=torch.float32, device=data_in.device).contiguous()
        for b in range(data_in.shape[0]):
            data_out[b][face_index_map[b] >= 0] = data_in[b][face_index_map[b] >= 0]

        ctx.save_for_backward(data_in, face_index_map)
        return data_out

    @staticmethod
    def backward(ctx, gradients):
        data_in, face_index_map = ctx.saved_tensors
        face_index_map = face_index_map.contiguous()
        grad_out = gradients.contiguous()
        grad_in = torch.zeros(grad_out.shape, dtype=torch.float32, device=grad_out.device).contiguous()
        for b in range(grad_in.shape[0]):
            grad_in[b][face_index_map[b] >= 0] = grad_out[b][face_index_map[b] >= 0]

        return grad_in, None


def mask_foreground(data, face_index_map):
    return MaskForeground().apply(data, face_index_map)
