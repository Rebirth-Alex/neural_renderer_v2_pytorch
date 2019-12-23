import torch
import torch.nn.functional as F
from neural_renderer_torch.cuda.rasterize_cuda import face_index_map_forward_safe, \
    face_index_map_forward_unsafe, compute_weight_map_c

from . import lights as light_lib
from .rasterize_param import RasterizeParam, RasterizeHyperparam
from .utils import pad_zeros, maximum, to_map, mask_foreground


#######################################################################################################################
# Core functions


class FaceIndexMap(torch.nn.Module):
    def __init__(self, num_faces, image_size, near, far, draw_backside):
        super(FaceIndexMap, self).__init__()
        self.num_faces = num_faces
        self.image_size = image_size
        self.near = near
        self.far = far
        self.draw_backside = draw_backside

    def forward(self, inputs, **kwargs):
        return self.forward_gpu_safe(inputs)
        # return self.forward_gpu_unsafe(inputs)

    def forward_gpu_safe(self, inputs):
        faces = inputs
        batch_size, num_faces = faces.shape[:2]
        faces = faces.contiguous()

        face_index = torch.zeros((batch_size, self.image_size, self.image_size), dtype=torch.int32).reshape((-1,)) - 1
        face_index = face_index.to(faces.device)
        face_index_map = face_index_map_forward_safe(faces, face_index, self.num_faces, self.image_size,
                                                     self.near, self.far, int(self.draw_backside), 1e-8, 1e-4)
        face_index_map = face_index_map.reshape((batch_size, self.image_size, self.image_size))

        return face_index_map

    def forward_gpu_unsafe(self, inputs):
        faces = inputs[0]
        batch_size, num_faces = faces.shape[:2]
        faces = faces.contiguous()

        face_index = torch.zeros((batch_size, self.image_size, self.image_size), dtype=torch.int32).reshape((-1,)) - 1
        depth_map = torch.zeros((batch_size, self.image_size, self.image_size), dtype=torch.float32) * self.far + 1
        lock = torch.zeros((batch_size, self.image_size, self.image_size), dtype=torch.int32)

        face_index_map = face_index_map_forward_unsafe(faces, face_index, depth_map, lock,
                                                       self.num_faces, self.image_size, self.near, self.far,
                                                       int(self.draw_backside), 1e-8)
        face_index_map = face_index_map.reshape((batch_size, self.image_size, self.image_size))

        return face_index_map

    def backward(self, ctx, *grad_outputs):
        pass


def compute_face_index_map(faces, image_size, near, far, draw_backside):
    batch_size, num_faces = faces.shape[:2]
    fim = FaceIndexMap(num_faces, image_size, near, far, draw_backside)
    face_index_map = fim(faces)
    return face_index_map


def compute_weight_map(faces, face_index_map):
    batch_size, num_faces = faces.shape[:2]
    image_size = face_index_map.shape[1]

    weight_map = torch.zeros((batch_size * image_size * image_size, 3), dtype=torch.float32).to(faces.device)
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

    textures = textures.permute((0, 2, 3, 1))  # [bs, h, w, 3]
    textures = torch.reshape(textures, (batch_size, texture_height * texture_width, 3))  # [bs, h * w, 3]
    faces_z_map = to_map(faces[:, :, :, 2], face_index_map)  # [bs, is, is, 3]
    vertices_textures_map = to_map(faces_textures, face_index_map)  # [bs, is, is, 3, 2]
    depth_map = 1. / (weight_map / faces_z_map).sum(-1)  # [bs, is, is]

    # -> [bs, is, is, 2]
    vertices_textures_map_original = vertices_textures_map.clone()
    vertices_textures_map = (
            weight_map[:, :, :, :, None] * vertices_textures_map / faces_z_map[:, :, :, :, None]).sum(-2)
    vertices_textures_map = vertices_textures_map * depth_map[:, :, :, None]  # [bs, is, is, 2]
    vertices_textures_map = torch.max(vertices_textures_map, vertices_textures_map_original.min(-2).values)
    vertices_textures_map = torch.min(vertices_textures_map, vertices_textures_map_original.max(-2).values - eps)
    vertices_textures_map = mask_foreground(vertices_textures_map, face_index_map)

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
    w1 = (y_c_f - y_f) * (x_c_f - x_f)  # [bs, is, is]
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
    m = torch.zeros((face_indices.shape[0], vertices.shape[1]), dtype=torch.float32, device=vertices.device)  # [nf, nv]
    m[torch.arange(m.shape[0]), face_indices[:, 0].cpu().numpy()] = 1
    m[torch.arange(m.shape[0]), face_indices[:, 1].cpu().numpy()] = 1
    m[torch.arange(m.shape[0]), face_indices[:, 2].cpu().numpy()] = 1
    n = n.permute((0, 2, 1))  # [bs, 3, nf]
    n = torch.reshape(n, (-1, n.shape[-1]))  # [bs * 3, nf]
    n = torch.matmul(n, m)
    n = torch.reshape(n, (faces.shape[0], 3, vertices.shape[1]))  # [bs, 3, nv]
    n = n.permute((0, 2, 1))
    n = F.normalize(n, dim=2)  # [bs, nv, 3]
    n = n[:, face_indices.cpu().numpy()]  # [bs, nv, 3, 3]

    normal_map = to_map(n, face_index_map)  # [bs, is, is, 3, 3]
    if smooth:
        normal_map = torch.sum(weight_map[:, :, :, :, None] * normal_map, dim=-2)
    else:
        normal_map = torch.mean(normal_map, dim=-2)
    return normal_map


########################################################################################################################
class Rasterize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coordinate_map, vertices, faces, face_indices, face_index_map, weight_map,
                params: RasterizeParam,
                hyperparams: RasterizeHyperparam):

        # -> [batch_size, 1, image_size, image_size]
        if hyperparams.draw_silhouettes or params.backgrounds is not None:
            silhouettes = (0 <= face_index_map).type(torch.float32)[:, :, :, None]
        else:
            silhouettes = None

        if hyperparams.draw_rgb:
            # -> [batch_size, num_faces, 3, 3]
            faces_textures = params.vertices_textures[:, params.faces_textures.cpu().numpy()]

            # -> [batch_size, image_size, image_size, 3]
            rgb_map = sample_textures(faces, faces_textures, params.textures,
                                      face_index_map, weight_map, hyperparams.eps)

            if params.lights is not None:
                normal_map = compute_normal_map(vertices, face_indices, faces, face_index_map, weight_map)
                color_weight_map = torch.as_tensor(
                    torch.zeros_like(normal_map, dtype=torch.float32, device=rgb_map.device))
                for light in params.lights:
                    light.to(rgb_map.device)
                    if isinstance(light, light_lib.AmbientLight):
                        color_weight_map += light.color[:, None, None, :].expand(color_weight_map.shape)
                    elif isinstance(light, light_lib.DirectionalLight):
                        # [bs, is, is]
                        intensity = torch.sum(-light.direction[:, None, None, :].expand(normal_map.shape) * normal_map,
                                              -1)
                        if light.backside:
                            intensity = torch.abs(intensity)
                        else:
                            intensity = torch.relu(intensity)
                        intensity = intensity[:, :, :, None].expand(color_weight_map.shape)
                        color = light.color[:, None, None, :].expand(color_weight_map.shape)
                        color_weight_map += intensity * color
                    elif isinstance(light, light_lib.SpecularLight):
                        # [bs, is, is]
                        direction_eye = torch.as_tensor([0, 0, 1], dtype=torch.float32, device=rgb_map.device)
                        intensity = torch.sum(-direction_eye[None, None, None, :] * normal_map, -1)
                        if light.backside:
                            intensity = torch.abs(intensity)
                        else:
                            intensity = torch.relu(intensity)
                        intensity = intensity ** light.alpha[:, None, None]
                        intensity = intensity[:, :, :, None].expand(color_weight_map.shape)
                        color = light.color[:, None, None, :].expand(color_weight_map.shape)
                        color_weight_map += intensity * color
                rgb_map *= color_weight_map

            # blend backgrounds
            if params.backgrounds is not None:
                backgrounds = params.backgrounds.permute((0, 2, 3, 1))
                rgb_map = blend_backgrounds(face_index_map, rgb_map, backgrounds)

        # -> [batch_size, 1, image_size, image_size]
        if hyperparams.draw_depth:
            depth_map = compute_depth_map(faces, face_index_map, weight_map)[:, :, :, None]

        # merge
        if hyperparams.draw_rgb and hyperparams.draw_silhouettes and hyperparams.draw_depth:
            images = torch.cat((rgb_map, silhouettes, depth_map), -1)
        elif hyperparams.draw_rgb and hyperparams.draw_silhouettes and not hyperparams.draw_depth:
            images = torch.cat((rgb_map, silhouettes), -1)
        elif hyperparams.draw_rgb and not hyperparams.draw_silhouettes and hyperparams.draw_depth:
            images = torch.cat((rgb_map, depth_map), -1)
        elif hyperparams.draw_rgb and not hyperparams.draw_silhouettes and not hyperparams.draw_depth:
            images = rgb_map
        elif not hyperparams.draw_rgb and hyperparams.draw_silhouettes and hyperparams.draw_depth:
            images = torch.cat((silhouettes, depth_map), -1)
        elif not hyperparams.draw_rgb and hyperparams.draw_silhouettes and not hyperparams.draw_depth:
            images = silhouettes
        elif not hyperparams.draw_rgb and not hyperparams.draw_silhouettes and hyperparams.draw_depth:
            images = depth_map
        elif not hyperparams.draw_rgb and not hyperparams.draw_silhouettes and not hyperparams.draw_depth:
            raise Exception
        else:
            raise Exception

        ctx.save_for_backward(images)

        return images

    @staticmethod
    def backward(ctx, gradients):
        images, = ctx.saved_tensors
        grad_output = gradients
        batch_size, image_size, _, num_channels = images.shape
        step = 2. / image_size

        grad_y_r = -((images[:, :-1, :] - images[:, 1:, :]) * grad_output[:, 1:, :]).sum(-1) / step
        grad_y_r = pad_zeros(grad_y_r[:, :, :, None], 1, 1, 'right') + pad_zeros(grad_y_r[:, :, :, None], 1, 1, 'left')
        grad_y_l = -((images[:, 1:, :] - images[:, :-1, :]) * grad_output[:, :-1, :]).sum(-1) / step
        grad_y_l = pad_zeros(grad_y_l[:, :, :, None], 1, 1, 'left') + pad_zeros(grad_y_l[:, :, :, None], 1, 1, 'right')
        grad_y = maximum(grad_y_r, grad_y_l)

        grad_x_r = -((images[:, :, :-1] - images[:, :, 1:]) * grad_output[:, :, 1:]).sum(-1) / step
        grad_x_r = pad_zeros(grad_x_r[:, :, :, None], 1, 2, 'right') + pad_zeros(grad_x_r[:, :, :, None], 1, 2, 'left')
        grad_x_l = -((images[:, :, 1:] - images[:, :, :-1]) * grad_output[:, :, :-1]).sum(-1) / step
        grad_x_l = pad_zeros(grad_x_l[:, :, :, None], 1, 2, 'left') + pad_zeros(grad_x_l[:, :, :, None], 1, 2, 'right')
        grad_x = maximum(grad_x_r, grad_x_l)

        grad_loss_xy = torch.cat((grad_x, grad_y), -1)

        # faces: [bs, nf, 3, 3]
        # face_index_map: [bs, is, is]
        # weight_map: [bs, is, is, 3]
        return grad_loss_xy, None, None, None, None, None, None, None


# Interfaces
def rasterize_all(
        vertices,
        faces,
        params: RasterizeParam,
        hyperparams: RasterizeHyperparam):
    # vertices: [batch_size, num_vertices, 3]
    # faces: [num_faces, 3]
    # vertices_textures: [batch_size, num_vertices_textures, 2]
    # faces_textures: [num_faces, 3]
    # textures: [batch_size, 3, height, width]

    assert vertices.ndim == 3
    assert vertices.shape[2] == 3
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    if hyperparams.draw_rgb:
        assert params.vertices_textures.ndim == 3
        assert params.vertices_textures.shape[2] == 2
        assert params.faces_textures.ndim == 2
        assert params.faces_textures.shape[1] == 3
        assert params.textures.ndim == 4
        assert params.textures.shape[1] == 3
    if params.background_color is not None:
        if hyperparams.anti_aliasing:
            backgrounds = torch.zeros((vertices.shape[0], 3, hyperparams.image_size * 2, hyperparams.image_size * 2),
                                      dtype=torch.float32, device=vertices.device)
        else:
            backgrounds = torch.zeros((vertices.shape[0], 3, hyperparams.image_size, hyperparams.image_size),
                                      dtype=torch.float32, device=vertices.device)
        backgrounds = backgrounds * torch.as_tensor(params.background_color)[None, :, None, None]
    elif params.backgrounds is not None:
        assert params.backgrounds.ndim == 4
        assert params.backgrounds.shape[0] == vertices.shape[0]
        assert params.backgrounds.shape[1] == 3
        if hyperparams.anti_aliasing:
            assert params.backgrounds.shape[2] == hyperparams.image_size * 2
            assert params.backgrounds.shape[3] == hyperparams.image_size * 2
        else:
            assert params.backgrounds.shape[2] == hyperparams.image_size
            assert params.backgrounds.shape[3] == hyperparams.image_size

    if hyperparams.anti_aliasing:
        hyperparams.image_size *= 2

    # -> [batch_size, num_faces, 3, 3]
    face_indices = faces.clone()
    faces = vertices[:, faces.cpu().numpy()]

    # -> [batch_size, num_faces, 3, 3]
    face_index_map = compute_face_index_map(faces,
                                            hyperparams.image_size, hyperparams.near,
                                            hyperparams.far, hyperparams.draw_backside)

    # -> [batch_size, image_size, image_size, 3]
    weight_map = compute_weight_map(faces, face_index_map)

    faces_map = to_map(faces[:, :, :, :2], face_index_map)  # [bs, is, is, 3, 3]
    coordinate_map = torch.sum(faces_map * weight_map[:, :, :, :, None], -2)

    r = Rasterize()
    images = r.apply(coordinate_map, vertices, faces, face_indices, face_index_map, weight_map, params, hyperparams)

    # -> [batch_size, 3, image_size, image_size]
    images = images.permute((0, 3, 1, 2))
    images = torch.flip(images, dims=(2, 3))
    # Flip copies whole array, hence, there is a performance impact
    # However, PyTorch is not planning to implement index style flipping (i.e. ::-1)

    # down sampling
    if hyperparams.anti_aliasing:
        # average pooling. faster than cf.average_pooling_2d(images, 2, 2)
        images = (
                images[:, :, 0::2, 0::2] +
                images[:, :, 1::2, 0::2] +
                images[:, :, 0::2, 1::2] +
                images[:, :, 1::2, 1::2])
        images /= 4.
    return images


def rasterize_silhouettes(vertices, faces, params: RasterizeParam, hyperparams: RasterizeHyperparam):
    hyperparams.draw_rgb = False
    hyperparams.draw_silhouettes = True
    hyperparams.draw_depth = False

    images = rasterize_all(vertices=vertices, faces=faces, params=params, hyperparams=hyperparams)
    return images[:, 0]


def rasterize_rgba(vertices, faces, params: RasterizeParam, hyperparams: RasterizeHyperparam):
    hyperparams.draw_rgb = True
    hyperparams.draw_silhouettes = True
    hyperparams.draw_depth = False

    images = rasterize_all(vertices=vertices, faces=faces, params=params, hyperparams=hyperparams)
    return images


def rasterize_rgb(vertices, faces, params: RasterizeParam, hyperparams: RasterizeHyperparam):
    hyperparams.draw_rgb = True
    hyperparams.draw_silhouettes = False
    hyperparams.draw_depth = False

    images = rasterize_all(vertices=vertices, faces=faces, params=params, hyperparams=hyperparams)
    return images


def rasterize_depth(vertices, faces, params: RasterizeParam, hyperparams: RasterizeHyperparam):
    hyperparams.draw_rgb = False
    hyperparams.draw_silhouettes = False
    hyperparams.draw_depth = True

    images = rasterize_all(vertices=vertices, faces=faces, params=params, hyperparams=hyperparams)
    return images[:, 0]
