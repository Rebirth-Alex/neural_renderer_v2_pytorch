import functools

import torch
import torch.nn.functional as F
from neural_renderer.cuda.rasterize_cuda import tomap_forward, tomap_backward, \
    mask_foreground_forward, mask_foreground_backward, face_index_map_forward_safe, \
    face_index_map_forward_unsafe, compute_weight_map_c
from torch.autograd import Function

from . import differentiation
from . import lights as light_lib

########################################################################################################################
# Parameters

# camera
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100

# rendering
DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_DRAW_BACKSIDE = True

# others
DEFAULT_EPS = 1e-5


########################################################################################################################
# Utility functions


class ToMap(Function):
    """
    Test code:
    import chainer.gradient_check
    data_in = cp.random.randn(*(16, 128, 3, 5)).astype('float32')
    indices = cp.random.randint(-1, 128, size=(16, 8, 8)).astype('int32')
    grad_out = cp.random.randn(16, 8, 8, 3, 5).astype('float32')
    data_out = ToMap()(data_in, indices)
    for i1 in range(16):
        for i2 in range(8):
            for i3 in range(8):
                i4 = indices[i1, i2, i3]
                if i4 < 0:
                    chainer.testing.assert_allclose(data_out[i1, i2, i3].data, data_out[i1, i2, i3].data * 0)
                else:
                    chainer.testing.assert_allclose(data_out[i1, i2, i3].data, data_in[i1, i4])
    chainer.gradient_check.check_backward(ToMap(), (data_in, indices), grad_out, no_grads=(False, True), rtol=1e-2, atol=1e-03)
    """

    """
    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[1].dtype.kind == 'i',
            in_types[0].shape[0] == in_types[1].shape[0],
        )
    """

    def forward_gpu(self, inputs):
        # data_in: [bs, nf, ...]
        # indices: [bs, is, is]
        # data_out: [bs, is, is, ..]
        data_in, indices = [inp.contiguous() for inp in inputs]
        data_out = torch.zeros(tuple(list(indices.shape[:3]) + list(data_in.shape[2:])),
                               dtype=torch.float32).contiguous()

        tomap_forward(data_in=data_in,
                      indices=indices,
                      data_out=data_out,
                      image_size=indices.shape[1],
                      num_features=data_in.shape[1],
                      dim=functools.reduce(lambda x, y: x * y, data_in.shape[2:])
                      )
        return data_out,

    def backward_gpu(self, inputs, gradients):
        # data_in: [bs, nf, ...]
        # indices: [bs, is, is]
        # data_out: [bs, is, is, ..]
        data_in_shape = inputs[0].shape
        indices = inputs[1].contiguous()
        grad_out = gradients[0].contiguous()
        grad_in = torch.zeros(data_in_shape, dtype=torch.float32).contiguous()

        tomap_backward(grad_in=grad_in,
                       indices=indices,
                       grad_out=grad_out,
                       image_size=indices.shape[1],
                       num_features=data_in_shape[1],
                       dim=functools.reduce(lambda x, y: x * y, data_in_shape[2:]))
        return grad_in, None


def to_map(data, indices):
    return ToMap()(data, indices)


class MaskForeground(Function):
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

    def forward_gpu(self, inputs):
        data_in, face_index_map = [inp.contiguous() for inp in inputs]
        if data_in.ndimensions() == 3:
            dim = 1
        else:
            dim = functools.reduce(lambda x, y: x * y, data_in.shape[3:])
        data_out = torch.zeros(data_in.shape, dtype=torch.float32).contiguous()
        mask_foreground_forward(face_index_map, data_in, data_out, dim)

        return data_out,

    def backward_gpu(self, inputs, gradients):
        face_index_map = inputs[1].contiguous()
        grad_out = gradients[0].contiguous()
        grad_in = torch.zeros(grad_out.shape, dtype=torch.float32).contiguous()
        if grad_in.ndimension() == 3:
            dim = 1
        else:
            dim = functools.reduce(lambda x, y: x * y, grad_in.shape[3:])
        mask_foreground_backward(face_index_map, grad_in, grad_out, dim)

        return grad_in, None


def mask_foreground(data, face_index_map):
    return MaskForeground()(data, face_index_map)


########################################################################################################################
# Core functions


class FaceIndexMap(Function):
    def __init__(self, num_faces, image_size, near, far, draw_backside):
        self.num_faces = num_faces
        self.image_size = image_size
        self.near = near
        self.far = far
        self.draw_backside = draw_backside

    def forward_gpu(self, inputs):
        return self.forward_gpu_safe(inputs)
        # return self.forward_gpu_unsafe(inputs)

    def forward_gpu_safe(self, inputs):
        faces = inputs[0]
        batch_size, num_faces = faces.shape[:2]
        faces = faces.contiguous()

        loop = torch.arange(batch_size * self.image_size * self.image_size).type(torch.int32)
        face_index = torch.zeros_like(faces)
        face_index_map = face_index_map_forward_safe(faces, face_index, self.num_faces, self.image_size,
                                                     self.near, self.far, int(self.draw_backside), 1e-8)
        face_index_map = face_index_map.reshape((batch_size, self.image_size, self.image_size))

        return face_index_map,

    def forward_gpu_unsafe(self, inputs):
        faces = inputs[0]
        batch_size, num_faces = faces.shape[:2]
        faces = faces.contiguous()

        loop = torch.arange(batch_size * self.num_faces).type(torch.int32)
        face_index_map = torch.zeros((batch_size, self.image_size, self.image_size), dtype=torch.int32) - 1
        depth_map = torch.zeros((batch_size, self.image_size, self.image_size), dtype=torch.float32) * self.far + 1
        lock = torch.zeros((batch_size, self.image_size, self.image_size), dtype=torch.int32)

        face_index_map = face_index_map_forward_unsafe(faces, face_index_map, depth_map, lock,
                                                       self.num_faces, self.image_size, self.near, self.far,
                                                       int(self.draw_backside), 1e-8)
        face_index_map = face_index_map.reshape((batch_size, self.image_size, self.image_size))

        return face_index_map,


def compute_face_index_map(faces, image_size, near, far, draw_backside):
    batch_size, num_faces = faces.shape[:2]
    face_index_map = FaceIndexMap(num_faces, image_size, near, far, draw_backside)(faces).data
    return face_index_map


def compute_weight_map(faces, face_index_map):
    batch_size, num_faces = faces.shape[:2]
    image_size = face_index_map.shape[1]

    weight_map = torch.zeros((batch_size * image_size * image_size, 3), dtype=torch.float32)
    if isinstance(faces, torch.Tensor):
        faces = faces.data
    faces = faces.contiguous()
    face_index_map = face_index_map.contiguous()
    weight_map = weight_map.contiguous()
    compute_weight_map_c(faces, face_index_map.flatten(), weight_map, num_faces, image_size)
    weight_map = weight_map.reshape((batch_size, image_size, image_size, 3))
    return weight_map


def compute_depth_map(faces, face_index_map, weight_map):
    # faces: [bs, nf, 3, 3]
    # face_index_map: [bs, is, is]
    # weight_map: [bs, is, is, 3]

    faces_z_map = to_map(faces[:, :, :, -1:], face_index_map)[:, :, :, :, 0]  # [bs, is, is, 3]
    depth_map = 1. / torch.sum(weight_map / faces_z_map, -1)
    depth_map = mask_foreground(depth_map, face_index_map)
    return depth_map


def compute_silhouettes(face_index_map):
    return (0 <= face_index_map).astype('float32')


def compute_coordinate_map(faces, face_index_map, weight_map):
    # faces: [bs, nf, 3, 3]
    # face_index_map: [bs, is, is]
    # weight_map: [bs, is, is, 3]
    faces_map = to_map(faces[:, :, :, :2], face_index_map)  # [bs, is, is, 3, 3]
    coordinate_map = torch.sum(faces_map * weight_map[:, :, :, :, None], -2)
    return coordinate_map


def sample_textures(faces, faces_textures, textures, face_index_map, weight_map, eps):
    # faces: [bs, nf, 3, 3]
    # faces_textures: [bs, nf, 3, 2]
    # textures: [bs, 3, is, is]
    # face_index_map: [bs, is, is]
    # weight_map: [bs, is, is, 3]
    batch_size, num_faces = faces.shape[:2]
    texture_height, texture_width = textures.shape[2:]
    if isinstance(faces, torch.Tensor):
        faces = faces.data.numpy()
    if isinstance(faces_textures, torch.Tensor):
        faces_textures = faces_textures.data.numpy()
    if isinstance(face_index_map, torch.Tensor):
        face_index_map = face_index_map.data.numpy()
    if isinstance(weight_map, torch.Tensor):
        weight_map = weight_map.data

    textures = textures.permute((0, 2, 3, 1))  # [bs, h, w, 3]
    textures = torch.reshape(textures, (batch_size, texture_height * texture_width, 3))  # [bs, h * w, 3]
    faces_z_map = to_map(faces[:, :, :, 2], face_index_map).data  # [bs, is, is, 3]
    vertices_textures_map = to_map(faces_textures, face_index_map).data  # [bs, is, is, 3, 2]
    depth_map = 1. / (weight_map / faces_z_map).sum(-1)  # [bs, is, is]

    # -> [bs, is, is, 2]
    vertices_textures_map_original = vertices_textures_map.copy()
    vertices_textures_map = (
            weight_map[:, :, :, :, None] * vertices_textures_map / faces_z_map[:, :, :, :, None]).sum(-2)
    vertices_textures_map = vertices_textures_map * depth_map[:, :, :, None]  # [bs, is, is, 2]
    vertices_textures_map = torch.max(vertices_textures_map, vertices_textures_map_original.min(-2))
    vertices_textures_map = torch.min(vertices_textures_map, vertices_textures_map_original.max(-2) - eps)
    vertices_textures_map = mask_foreground(vertices_textures_map, face_index_map).data

    x_f = vertices_textures_map[:, :, :, 0]
    y_f = vertices_textures_map[:, :, :, 1]
    x_f_f = torch.floor(x_f)
    y_f_f = torch.floor(y_f)
    x_c_f = x_f_f + 1
    y_c_f = y_f_f + 1
    x_f_i = x_f_f.type(torch.int32)
    y_f_i = y_f_f.type(torch.int32)
    x_c_i = x_c_f.type(torch.int32)
    y_c_i = y_c_f.type(torch.int32)

    #
    vtm1 = (y_f_i * texture_width + x_f_i)  # [bs, is, is]
    vtm2 = (y_f_i * texture_width + x_c_i)  # [bs, is, is]
    vtm3 = (y_c_i * texture_width + x_f_i)  # [bs, is, is]
    vtm4 = (y_c_i * texture_width + x_c_i)  # [bs, is, is]
    w1 = (y_c_f - y_f) * (x_c_f - x_f)  # [bs * is * is]
    w2 = (y_c_f - y_f) * (x_f - x_f_f)  # [bs, is, is]
    w3 = (y_f - y_f_f) * (x_c_f - x_f)  # [bs, is, is]
    w4 = (y_f - y_f_f) * (x_f - x_f_f)  # [bs, is, is]
    images = (
            w1[:, :, :, None] * to_map(textures, vtm1) +
            w2[:, :, :, None] * to_map(textures, vtm2) +
            w3[:, :, :, None] * to_map(textures, vtm3) +
            w4[:, :, :, None] * to_map(textures, vtm4))

    # mask foreground
    images = mask_foreground(images, face_index_map)

    return images


def blend_backgrounds(face_index_map, rgb_map, backgrounds):
    foreground_map = (0 <= face_index_map).astype('float32')[:, :, :, None]  # [bs, is, is, 1]
    rgb_map = foreground_map * rgb_map + (1 - foreground_map) * backgrounds[:, ::-1, ::-1]
    return rgb_map


def compute_normal_map(vertices, face_indices, faces, face_index_map, weight_map, smooth=True):
    # faces: [bs, nf, 3, 3]
    # face_index_map: [bs, is, is]
    # weight_map: [bs, is, is, 3]

    v01 = faces[:, :, 1, :] - faces[:, :, 0, :]
    v12 = faces[:, :, 2, :] - faces[:, :, 1, :]
    v01 = torch.reshape(v01, (-1, 3))
    v12 = torch.reshape(v12, (-1, 3))
    n = torch.cross(v01, v12)
    n = torch.reshape(n, (faces.shape[0], faces.shape[1], 3))  # [bs, nf, 3]
    m = torch.zeros((face_indices.shape[0], vertices.shape[1]), dtype=torch.float32)  # [nf, nv]
    m[torch.arange(m.shape[0]), face_indices[:, 0]] = 1
    m[torch.arange(m.shape[0]), face_indices[:, 1]] = 1
    m[torch.arange(m.shape[0]), face_indices[:, 2]] = 1
    n = n.transpose((0, 2, 1))  # [bs, 3, nf]
    n = torch.reshape(n, (-1, n.shape[-1]))  # [bs * 3, nf]
    n = torch.matmul(n, m)
    n = torch.reshape(n, (faces.shape[0], 3, vertices.shape[1]))  # [bs, 3, nv]
    n = n.transpose((0, 2, 1))
    n = F.normalize(n, axis=2)  # [bs, nv, 3]
    n = n[:, face_indices]  # [bs, nv, 3, 3]

    normal_map = to_map(n, face_index_map)  # [bs, is, is, 3, 3]
    if smooth:
        normal_map = torch.sum(weight_map[:, :, :, :, None] * normal_map, dim=-2)
    else:
        normal_map = torch.mean(normal_map, dim=-2)
    return normal_map


########################################################################################################################
# Interfaces
def rasterize_all(
        vertices,
        faces,
        vertices_textures=None,
        faces_textures=None,
        textures=None,
        background_color=None,
        backgrounds=None,
        lights=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
        draw_rgb=True,
        draw_silhouettes=True,
        draw_depth=True,
):
    # vertices: [batch_size, num_vertices, 3]
    # faces: [num_faces, 3]
    # vertices_textures: [batch_size, num_vertices_textures, 2]
    # faces_textures: [num_faces, 3]
    # textures: [batch_size, 3, height, width]
    assert vertices.ndim == 3
    assert vertices.shape[2] == 3
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    if draw_rgb:
        assert vertices_textures.ndim == 3
        assert vertices_textures.shape[2] == 2
        assert faces_textures.ndim == 2
        assert faces_textures.shape[1] == 3
        assert textures.ndim == 4
        assert textures.shape[1] == 3
    if background_color is not None:
        if anti_aliasing:
            backgrounds = torch.zeros((vertices.shape[0], 3, image_size * 2, image_size * 2), dtype=torch.float32)
        else:
            backgrounds = torch.zeros((vertices.shape[0], 3, image_size, image_size), dtype=torch.float32)
        backgrounds = backgrounds * torch.as_tensor(background_color)[None, :, None, None]
    elif backgrounds is not None:
        assert backgrounds.ndim == 4
        assert backgrounds.shape[0] == vertices.shape[0]
        assert backgrounds.shape[1] == 3
        if anti_aliasing:
            assert backgrounds.shape[2] == image_size * 2
            assert backgrounds.shape[3] == image_size * 2
        else:
            assert backgrounds.shape[2] == image_size
            assert backgrounds.shape[3] == image_size

    if anti_aliasing:
        image_size *= 2

    # -> [batch_size, num_faces, 3, 3]
    face_indices = faces.copy()
    faces = vertices[:, faces]

    # -> [batch_size, num_faces, 3, 3]
    face_index_map = compute_face_index_map(faces, image_size, near, far, draw_backside)

    # -> [batch_size, image_size, image_size, 3]
    weight_map = compute_weight_map(faces, face_index_map)

    # -> [batch_size, 1, image_size, image_size]
    if draw_silhouettes or backgrounds is not None:
        silhouettes = compute_silhouettes(face_index_map)[:, :, :, None]

    if draw_rgb:
        # -> [batch_size, num_faces, 3, 3]
        faces_textures = vertices_textures[:, faces_textures]

        # -> [batch_size, image_size, image_size, 3]
        rgb_map = sample_textures(faces, faces_textures, textures, face_index_map, weight_map, eps)

        if lights is not None:
            normal_map = compute_normal_map(vertices, face_indices, faces, face_index_map, weight_map)
            color_weight_map = torch.as_tensor(torch.zeros_like(normal_map, dtype=torch.float32))
            for light in lights:
                if isinstance(light, light_lib.AmbientLight):
                    color_weight_map += light.color[:, None, None, :].expand(color_weight_map.shape)
                if isinstance(light, light_lib.DirectionalLight):
                    # [bs, is, is]
                    intensity = torch.sum(-light.direction[:, None, None, :].expand(normal_map.shape) * normal_map, -1)
                    if light.backside:
                        intensity = torch.abs(intensity)
                    else:
                        intensity = torch.relu(intensity)
                    intensity = intensity[:, :, :, None].expand(color_weight_map.shape)
                    color = light.color[:, None, None, :].expand(color_weight_map.shape)
                    color_weight_map += intensity * color
                if isinstance(light, light_lib.SpecularLight):
                    # [bs, is, is]
                    direction_eye = torch.as_tensor([0, 0, 1], dtype=torch.float32)
                    intensity = torch.sum(-direction_eye[None, None, None, :] * normal_map, -1)
                    if light.backside:
                        intensity = torch.abs(intensity)
                    else:
                        intensity = torch.relu(intensity)
                    intensity **= light.alpha[:, None, None]
                    intensity = intensity[:, :, :, None].expand(color_weight_map.shape)
                    color = light.color[:, None, None, :].expand(color_weight_map.shape)
                    color_weight_map += intensity * color
            rgb_map *= color_weight_map

        # blend backgrounds
        if backgrounds is not None:
            backgrounds = backgrounds.transpose((0, 2, 3, 1))
            rgb_map = blend_backgrounds(face_index_map, rgb_map, backgrounds)

    # -> [batch_size, 1, image_size, image_size]
    if draw_depth:
        depth_map = compute_depth_map(faces, face_index_map, weight_map)[:, :, :, None]

    # merge
    if draw_rgb and draw_silhouettes and draw_depth:
        images = torch.cat((rgb_map, silhouettes, depth_map), -1)
    elif draw_rgb and draw_silhouettes and not draw_depth:
        images = torch.cat((rgb_map, silhouettes), -1)
    elif draw_rgb and not draw_silhouettes and draw_depth:
        images = torch.cat((rgb_map, depth_map), -1)
    elif draw_rgb and not draw_silhouettes and not draw_depth:
        images = rgb_map
    elif not draw_rgb and draw_silhouettes and draw_depth:
        images = torch.cat((silhouettes, depth_map), -1)
    elif not draw_rgb and draw_silhouettes and not draw_depth:
        images = silhouettes
    elif not draw_rgb and not draw_silhouettes and draw_depth:
        images = depth_map
    elif not draw_rgb and not draw_silhouettes and not draw_depth:
        raise Exception

    # -> [batch_size, image_size, image_size, 2]
    coordinate_map = compute_coordinate_map(faces, face_index_map, weight_map)

    # -> [batch_size, 3, image_size, image_size]
    images = differentiation.differentiation(images, coordinate_map)
    images = images[:, ::-1, ::-1, :].transpose((0, 3, 1, 2))

    # down sampling
    if anti_aliasing:
        # average pooling. faster than cf.average_pooling_2d(images, 2, 2)
        images = (
                images[:, :, 0::2, 0::2] +
                images[:, :, 1::2, 0::2] +
                images[:, :, 0::2, 1::2] +
                images[:, :, 1::2, 1::2])
        images /= 4.

    return images


def rasterize_silhouettes(
        vertices,
        faces,
        background_color=None,
        backgrounds=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
):
    images = rasterize_all(
        vertices=vertices,
        faces=faces,
        background_color=background_color,
        backgrounds=backgrounds,
        image_size=image_size,
        near=near,
        far=far,
        eps=eps,
        anti_aliasing=anti_aliasing,
        draw_backside=draw_backside,
        draw_rgb=False,
        draw_silhouettes=True,
        draw_depth=False,
    )
    return images[:, 0]


def rasterize_rgba(
        vertices,
        faces,
        vertices_textures,
        faces_textures,
        textures,
        background_color=None,
        backgrounds=None,
        lights=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
):
    images = rasterize_all(
        vertices=vertices,
        faces=faces,
        vertices_textures=vertices_textures,
        faces_textures=faces_textures,
        textures=textures,
        background_color=background_color,
        backgrounds=backgrounds,
        lights=lights,
        image_size=image_size,
        near=near,
        far=far,
        eps=eps,
        anti_aliasing=anti_aliasing,
        draw_backside=draw_backside,
        draw_rgb=True,
        draw_silhouettes=True,
        draw_depth=False,
    )
    return images


def rasterize_rgb(
        vertices,
        faces,
        vertices_textures,
        faces_textures,
        textures,
        background_color=None,
        backgrounds=None,
        lights=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
):
    images = rasterize_all(
        vertices=vertices,
        faces=faces,
        vertices_textures=vertices_textures,
        faces_textures=faces_textures,
        textures=textures,
        background_color=background_color,
        backgrounds=backgrounds,
        lights=lights,
        image_size=image_size,
        near=near,
        far=far,
        eps=eps,
        anti_aliasing=anti_aliasing,
        draw_backside=draw_backside,
        draw_rgb=True,
        draw_silhouettes=False,
        draw_depth=False,
    )
    return images


def rasterize(
        vertices,
        faces,
        vertices_textures,
        faces_textures,
        textures,
        background_color=None,
        backgrounds=None,
        lights=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
):
    images = rasterize_all(
        vertices=vertices,
        faces=faces,
        vertices_textures=vertices_textures,
        faces_textures=faces_textures,
        textures=textures,
        background_color=background_color,
        backgrounds=backgrounds,
        lights=lights,
        image_size=image_size,
        near=near,
        far=far,
        eps=eps,
        anti_aliasing=anti_aliasing,
        draw_backside=draw_backside,
        draw_rgb=True,
        draw_silhouettes=False,
        draw_depth=False,
    )
    return images


def rasterize_depth(
        vertices,
        faces,
        backgrounds=None,
        background_color=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
):
    images = rasterize_all(
        vertices=vertices,
        faces=faces,
        background_color=background_color,
        backgrounds=backgrounds,
        image_size=image_size,
        near=near,
        far=far,
        eps=eps,
        anti_aliasing=anti_aliasing,
        draw_backside=draw_backside,
        draw_rgb=False,
        draw_silhouettes=False,
        draw_depth=True,
    )
    return images[:, 0]
