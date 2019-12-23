import math

import neural_renderer_torch
from neural_renderer_torch.rasterize_param import RasterizeParam, RasterizeHyperparam


class Renderer(object):
    def __init__(self):
        # rendering
        self.image_size = 256
        self.anti_aliasing = True
        self.draw_backside = True
        self.background_color = None

        # camera
        self.perspective = True
        self.viewing_angle = 30
        self.viewpoints = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]
        self.camera_mode = 'look_at'
        self.camera_direction = [0, 0, 1]
        self.near = 0.1
        self.far = 100

    def transform_vertices(self, vertices, lights=None):
        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = neural_renderer_torch.look_at(vertices, self.viewpoints)
        elif self.camera_mode == 'look':
            vertices = neural_renderer_torch.look(vertices, self.viewpoints, self.camera_direction)

        # perspective transformation
        if self.perspective:
            vertices = neural_renderer_torch.perspective(vertices, angle=self.viewing_angle)

        return vertices

    def render_silhouettes(self, vertices, faces, backgrounds=None):
        vertices = self.transform_vertices(vertices)

        params = RasterizeParam(background_color=self.background_color, backgrounds=backgrounds)
        hyperparams = RasterizeHyperparam(image_size=self.image_size, near=self.near,
                                          far=self.far, anti_aliasing=self.anti_aliasing,
                                          draw_backside=self.draw_backside)

        images = neural_renderer_torch.rasterize_silhouettes(vertices, faces, params, hyperparams)
        return images

    def render(self, vertices, faces, vertices_t, faces_t, textures, backgrounds=None, lights=None):
        vertices = self.transform_vertices(vertices)
        params = RasterizeParam(vertices_textures=vertices_t, faces_textures=faces_t, textures=textures,
                                background_color=self.background_color, backgrounds=backgrounds, lights=lights)
        hyperparams = RasterizeHyperparam(image_size=self.image_size, near=self.near, far=self.far,
                                          anti_aliasing=self.anti_aliasing, draw_backside=self.draw_backside)

        images = neural_renderer_torch.rasterize_rgba(vertices, faces, params, hyperparams)
        return images

    def render_rgb(self, vertices, faces, vertices_t, faces_t, textures, backgrounds=None, lights=None):
        vertices = self.transform_vertices(vertices, lights)
        params = RasterizeParam(vertices_textures=vertices_t, faces_textures=faces_t, textures=textures,
                                background_color=self.background_color, backgrounds=backgrounds, lights=lights)
        hyperparams = RasterizeHyperparam(image_size=self.image_size, near=self.near, far=self.far,
                                          anti_aliasing=self.anti_aliasing, draw_backside=self.draw_backside)

        images = neural_renderer_torch.rasterize_rgb(vertices, faces, params, hyperparams)
        return images

    def render_depth(self, vertices, faces, backgrounds=None):
        vertices = self.transform_vertices(vertices)
        params = RasterizeParam(background_color=self.background_color, backgrounds=backgrounds)
        hyperparams = RasterizeHyperparam(image_size=self.image_size, near=self.near, far=self.far,
                                          anti_aliasing=self.anti_aliasing, draw_backside=self.draw_backside)

        images = neural_renderer_torch.rasterize_depth(vertices, faces, params, hyperparams)
        return images
