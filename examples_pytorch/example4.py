"""
Example 4. Finding camera parameters.
"""
import argparse
import os

import numpy as np
import torch
import tqdm
from PIL import Image

import neural_renderer_torch
from neural_renderer_torch.utils import make_gif


class Model(torch.nn.Module):
    def __init__(self, input_obj_file, input_ref_file=None):
        super(Model, self).__init__()

        # Load the OBJ file.
        vertices, faces = neural_renderer_torch.load_obj(input_obj_file)
        self.vertices = torch.as_tensor(vertices[None, :, :])
        self.faces = torch.as_tensor(faces)

        # Load the reference image.
        if input_ref_file is not None:
            self.image_ref = torch.as_tensor(neural_renderer_torch.imread(input_ref_file))
        else:
            self.image_ref = None

        # Set up the camera parameters.
        self.camera_position = torch.nn.Parameter(torch.tensor([6, 10, -14], dtype=torch.float32))

        # Set up the renderer.
        renderer = neural_renderer_torch.Renderer()
        renderer.viewpoints = self.camera_position
        self.renderer = renderer

    def to(self, device=None):
        super(Model, self).to(device)
        self.faces = self.faces.to(device)
        self.vertices = self.vertices.to(device)
        if self.image_ref is not None:
            self.image_ref = self.image_ref.to(device)

    def __call__(self):
        image = self.renderer.render_silhouettes(self.vertices, self.faces)
        loss = torch.sum(torch.pow(image - self.image_ref[None, :, :], 2))
        return loss


def make_ref_image(input_ref_file, input_obj_file, gpu):
    model = Model(input_obj_file)
    model.to(gpu)

    model.renderer.viewpoints = neural_renderer_torch.get_points_from_angles(2.732, 30, -15)
    images = model.renderer.render_silhouettes(model.vertices, model.faces)
    image = images.detach()[0].cpu().numpy()
    min_val, max_val = image.min(), image.max()
    image = (image - min_val) / (max_val - min_val) * 255

    image = Image.fromarray(image.astype(np.uint8))
    image.save(input_ref_file)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--input_obj_file', type=str, default='./examples_pytorch/data/teapot.obj')
    parser.add_argument('-ir', '--input_ref_file', type=str, default='./examples_pytorch/data/example4_ref.png')
    parser.add_argument('-or', '--output_res_file', type=str, default='./examples_pytorch/data/example4_res.gif')
    parser.add_argument('-mr', '--make_ref_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    return args


def run():
    args = parse_arguments()
    working_dir = os.path.dirname(args.output_res_file)

    # Currently, only .obj files are supported.
    if not args.input_obj_file.endswith('.obj'):
        raise RuntimeError('Only .obj files are currently supported as input.')

    if args.make_ref_image:
        make_ref_image(args.input_ref_file, args.input_obj_file)

    model = Model(args.input_obj_file, args.input_ref_file)
    model.to(args.gpu)

    # Render an initial image.
    images = model.renderer.render_silhouettes(model.vertices, model.faces)
    image = images.detach()[0].cpu().numpy()
    min_val, max_val = image.min(), image.max()
    image = (image - min_val) / (max_val - min_val) * 255
    image = Image.fromarray(image.astype(np.uint8))
    image.save('%s/example4_init.png' % working_dir)

    # Run the optimization loop.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Run the optimization loop.
    loop = tqdm.tqdm(range(1000))
    for i in loop:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        # Scale each frame to the [0, 255] interval.
        images = model.renderer.render_silhouettes(model.vertices, model.faces)
        image = images.detach()[0].cpu().numpy()
        min_val, max_val = image.min(), image.max()
        image = (image - min_val) / (max_val - min_val) * 255

        # Save each frame to the working directory.
        image = Image.fromarray(image.astype(np.uint8))
        image.save('%s/_tmp_%04d.png' % (working_dir, i))

        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        if loss.data < 70:
            break

    make_gif(working_dir, args.output_res_file)


if __name__ == '__main__':
    run()
