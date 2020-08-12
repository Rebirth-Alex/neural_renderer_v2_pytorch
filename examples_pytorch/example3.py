"""
Example 3. Optimizing textures.
"""
import argparse
import os

import numpy as np
import torch
import tqdm
from PIL import Image

import neural_renderer_torch as neural_renderer
from neural_renderer_torch.utils import make_gif

CAMERA_DISTANCE = 2.732
ELEVATION = 0


class Model(torch.nn.Module):
    def __init__(self, input_obj_file, input_ref_file):
        super(Model, self).__init__()

        # Load the OBJ file.
        vertices, faces = neural_renderer.load_obj(input_obj_file)
        self.vertices = torch.as_tensor(vertices[None, :, :])
        self.faces = torch.as_tensor(faces)

        # Create the textures.
        vertices_t, faces_t, textures = neural_renderer.create_textures(self.faces.shape[0], texture_size=4)
        self.vertices_t = torch.as_tensor(vertices_t[None, :, :])
        self.faces_t = torch.as_tensor(faces_t)
        self.textures = torch.nn.Parameter(torch.as_tensor(textures[None, :, :, :]), requires_grad=True)

        # Load the reference image.
        self.image_ref = neural_renderer.imread(input_ref_file)
        self.image_ref = torch.as_tensor(self.image_ref)

        # Set up the renderer.
        renderer = neural_renderer.Renderer()
        renderer.perspective = False
        self.renderer = renderer

    def to(self, device=None):
        super(Model, self).to(device)
        self.faces = self.faces.to(device)
        self.vertices = self.vertices.to(device)
        self.vertices_t = self.vertices_t.to(device)
        self.faces_t = self.faces_t.to(device)
        self.image_ref = self.image_ref.to(device)

    def __call__(self, textures):
        self.renderer.viewpoints = neural_renderer.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        images = self.renderer.render_rgb(
            self.vertices, self.faces, self.vertices_t, self.faces_t, torch.tanh(textures))
        loss = torch.sum((images[0] - self.image_ref.permute((2, 0, 1))) ** 2)
        return loss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--input_obj_file', type=str, default='./examples_pytorch/data/teapot.obj')
    parser.add_argument('-ir', '--input_ref_file', type=str, default='./examples_pytorch/data/example3_ref.png')
    parser.add_argument('-or', '--output_res_file', type=str, default='./examples_pytorch/data/example3_res.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    return args


def run():
    args = parse_arguments()
    working_dir = os.path.dirname(args.output_res_file)

    # Currently, only .obj files are supported.
    if not args.input_obj_file.endswith('.obj'):
        raise RuntimeError('Only .obj files are currently supported as input.')

    model = Model(args.input_obj_file, args.input_ref_file)
    model.to(args.gpu)

    # Create the optimizer object.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Run the optimization loop.
    loop = tqdm.tqdm(range(300))
    for _ in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        loss = model(model.textures)
        loss.backward()
        optimizer.step()

    # Run the optimization loop.
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Rendering')
        model.renderer.viewpoints = neural_renderer.get_points_from_angles(CAMERA_DISTANCE, ELEVATION, azimuth)

        images = model.renderer.render_rgb(
            model.vertices, model.faces, model.vertices_t, model.faces_t, torch.tanh(model.textures))
        image = images.detach()[0].cpu().numpy().transpose((1, 2, 0))
        image = np.maximum(image, 0) * 255  # Crop negative values

        # Save each frame to the working directory.
        image = Image.fromarray(image.astype(np.uint8))
        image.save('%s/_tmp_%04d.png' % (working_dir, num))

    make_gif(working_dir, args.output_res_file)


if __name__ == '__main__':
    run()
