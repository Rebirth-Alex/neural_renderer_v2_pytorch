DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100.0

# rendering
DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_DRAW_BACKSIDE = True

# others
DEFAULT_EPS = 1e-5


class RasterizeHyperparam:
    def __init__(self,
                 image_size=DEFAULT_IMAGE_SIZE,
                 near=DEFAULT_NEAR,
                 far=DEFAULT_FAR,
                 eps=DEFAULT_EPS,
                 anti_aliasing=DEFAULT_ANTI_ALIASING,
                 draw_backside=DEFAULT_DRAW_BACKSIDE,
                 draw_rgb=True,
                 draw_silhouettes=True,
                 draw_depth=True
                 ):
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps
        self.anti_aliasing = anti_aliasing
        self.draw_backside = draw_backside
        self.draw_rgb = draw_rgb
        self.draw_silhouettes = draw_silhouettes
        self.draw_depth = draw_depth


class RasterizeParam:
    def __init__(self,
                 vertices_textures=None,
                 faces_textures=None,
                 textures=None,
                 background_color=None,
                 backgrounds=None,
                 lights=None,
                 ):
        self.vertices_textures = vertices_textures
        self.faces_textures = faces_textures
        self.textures = textures
        self.background_color = background_color
        self.backgrounds = backgrounds
        self.lights = lights
