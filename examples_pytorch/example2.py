import argparse
import os

import numpy as np
import torch
import tqdm
from PIL import Image

import neural_renderer_torch
from neural_renderer_torch.utils import make_gif, imread

CAMERA_DISTANCE = 2.732
ELEVATION = 0
AZIMUTH = 90


class Model(torch.nn.Module):
    def __init__(self, input_obj_file, input_ref_file):
        super(Model, self).__init__()

        # Load the OBJ file.
        vertices, faces = neural_renderer_torch.load_obj(input_obj_file)
        self.vertices = torch.nn.Parameter(torch.as_tensor(vertices[None, :, :]))
        self.faces = torch.as_tensor(faces)

        # Load the reference image.
        self.image_ref = torch.as_tensor(imread(input_ref_file).mean(-1))

        # Set up the renderer.
        renderer = neural_renderer_torch.Renderer()
        self.renderer = renderer

    def to(self, device=None):
        super(Model, self).to(device)
        self.faces = self.faces.to(device)
        self.image_ref = self.image_ref.to(device)

    def __call__(self):
        self.renderer.viewpoints = neural_renderer_torch.get_points_from_angles(
            CAMERA_DISTANCE, ELEVATION, AZIMUTH)
        image = self.renderer.render_silhouettes(self.vertices, self.faces)
        loss = torch.sum(torch.pow(image - self.image_ref[None, :, :], 2))
        return loss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--input_obj_file', type=str, default='./examples_pytorch/data/teapot.obj')
    parser.add_argument('-ir', '--input_ref_file', type=str, default='./examples_pytorch/data/example2_ref.png')
    parser.add_argument('-oo', '--output_opt_file', type=str, default='./examples_pytorch/data/example2_opt.gif')
    parser.add_argument('-or', '--output_res_file', type=str, default='./examples_pytorch/data/example2_res.gif')
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
    optimizer = torch.optim.Adam(model.parameters())

    # Run the optimization loop.
    loop = tqdm.tqdm(range(300))
    for i in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        # Scale each frame to the [0, 255] interval.
        image = model.renderer.render_silhouettes(model.vertices, model.faces)
        image = image.detach()[0].cpu().numpy()
        min_val, max_val = image.min(), image.max()
        image = (image - min_val) / (max_val - min_val) * 255

        # Save each frame to the working directory.
        image = Image.fromarray(image.astype(np.uint8))
        image.save('%s/_tmp_%04d.png' % (working_dir, i))

    make_gif(working_dir, args.output_opt_file)

    # Run the rendering loop.
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Rendering')
        model.renderer.viewpoints = neural_renderer_torch.get_points_from_angles(CAMERA_DISTANCE, ELEVATION, azimuth)

        # Scale each frame to the [0, 255] interval.
        image = model.renderer.render_silhouettes(model.vertices, model.faces)
        image = image.detach()[0].cpu().numpy()
        min_val, max_val = image.min(), image.max()
        image = (image - min_val) / (max_val - min_val) * 255

        # Save each frame to the working directory.
        image = Image.fromarray(image.astype(np.uint8))
        image.save('%s/_tmp_%04d.png' % (working_dir, num))

    make_gif(working_dir, args.output_res_file)


if __name__ == '__main__':
    run()
