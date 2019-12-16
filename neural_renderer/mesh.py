import torch
import torch.nn as nn

from neural_renderer import load_obj


class Mesh(nn.Module):
    def __init__(self, filename_obj, texture_size=4, normalization=True):
        super(Mesh, self).__init__()

        # load .obj
        vertices, faces = load_obj(filename_obj, normalization)
        self.vertices = torch.as_tensor(vertices)
        self.faces = faces
        self.num_vertices = self.vertices.shape[0]
        self.num_faces = self.faces.shape[0]

        # create textures
        shape = (self.num_faces, texture_size, texture_size, texture_size, 3)
        self.textures = torch.nn.Parameter(torch.nn.init.normal(torch.zeros(shape)))
        self.texture_size = texture_size

    def to(self, *args, **kwargs):
        super(Mesh, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self.faces = self.faces.to(device)

    def get_batch(self, batch_size):
        # broadcast for minibatch
        vertices = self.vertices.expand([batch_size] + list(self.vertices.shape))
        faces = self.faces.expand([batch_size] + list(self.faces.shape))
        textures = torch.sigmoid(self.textures.expand([batch_size] + list(self.textures.shape)))
        return vertices, faces, textures

    def set_lr(self, lr_vertices, lr_textures):
        self.vertices.lr = lr_vertices
        self.textures.lr = lr_textures
