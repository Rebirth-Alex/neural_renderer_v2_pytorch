"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import argparse
import os

import numpy as np
import torch
import tqdm
from PIL import Image

import neural_renderer_torch
from neural_renderer_torch.utils import make_gif

CAMERA_DISTANCE = 2.732
ELEVATION = 30


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, default='./examples_pytorch/data/teapot.obj')
    parser.add_argument('-o', '--output_file', type=str, default='./examples_pytorch/data/example1.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    return args


def run():
    args = parse_arguments()
    working_dir = os.path.dirname(args.output_file)

    # Currently, only .obj files are supported.
    if not args.input_file.endswith('.obj'):
        raise RuntimeError('Only .obj files are currently supported as input.')

    # Load the input data:
    #    vertices: [num_vertices, 3]
    #    faces: # [num_faces, 3]
    vertices, faces = neural_renderer_torch.load_obj(args.input_file)

    # Add a batch size of 1:
    #    vertices: [1, num_vertices, 3]
    vertices = vertices[None, :, :]

    # Upload the data to the GPU.
    device = torch.device('cuda:' + str(args.gpu))
    torch.cuda.set_device(device)

    vertices = torch.tensor(vertices, device=device)
    faces = torch.tensor(faces, device=device)

    # Create the renderer object.
    renderer = neural_renderer_torch.Renderer()

    # Run the rendering loop.
    loop = tqdm.tqdm(range(0, 360, 4))

    for num, azimuth in enumerate(loop):
        loop.set_description('Rendering')
        renderer.viewpoints = neural_renderer_torch.get_points_from_angles(
            CAMERA_DISTANCE, ELEVATION, azimuth)

        # Scale each frame to the [0, 255] interval.
        image = renderer.render_silhouettes(vertices, faces)[0].cpu().numpy()
        min_val, max_val = image.min(), image.max()
        image = (image - min_val) / (max_val - min_val) * 255

        # Save each frame to the working directory.
        image = Image.fromarray(image.astype(np.uint8))
        image.save('%s/_tmp_%04d.png' % (working_dir, num))

    make_gif(working_dir, args.output_file)


if __name__ == '__main__':
    run()
