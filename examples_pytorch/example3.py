"""
Example 3. Optimizing textures.
"""
import argparse
import glob
import os
import subprocess

import cv2
import numpy as np
import torch
import tqdm

import neural_renderer_torch as neural_renderer


class Model(torch.nn.Module):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()

        # load .obj
        vertices, faces = neural_renderer.load_obj(filename_obj)
        self.vertices = torch.as_tensor(vertices[None, :, :])
        self.faces = torch.as_tensor(faces)

        # create textures
        vertices_t, faces_t, textures = neural_renderer.create_textures(self.faces.shape[0], texture_size=4)
        self.vertices_t = torch.as_tensor(vertices_t[None, :, :])
        self.faces_t = torch.as_tensor(faces_t)
        self.textures = torch.nn.Parameter(torch.as_tensor(textures[None, :, :, :]), requires_grad=True)

        # load reference image
        self.image_ref = neural_renderer.imread(filename_ref)
        self.image_ref = torch.as_tensor(self.image_ref)

        # setup renderer
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


def make_gif(working_directory, filename):
    # generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/_tmp_*.png %s' % (options, working_directory, filename), shell=True)
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default='./examples_chainer/data/teapot.obj')
    parser.add_argument('-ir', '--filename_ref', type=str, default='./examples_chainer/data/example3_ref.png')
    parser.add_argument('-or', '--filename_output', type=str, default='./examples_chainer/data/example3_result.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    working_directory = os.path.dirname(args.filename_output)

    model = Model(args.filename_obj, args.filename_ref)
    model.to(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loop = tqdm.tqdm(range(300))
    for _ in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        loss = model(model.textures)
        loss.backward()
        optimizer.step()

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.viewpoints = neural_renderer.get_points_from_angles(2.732, 0, azimuth)
        images = model.renderer.render_rgb(
            model.vertices, model.faces, model.vertices_t, model.faces_t, torch.tanh(model.textures))
        image = images.cpu().detach().numpy()[0].transpose((1, 2, 0))
        image = np.maximum(image, 0) * 255  # Crop negative values
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('%s/_tmp_%04d.png' % (working_directory, num), image)
    make_gif(working_directory, args.filename_output)


if __name__ == '__main__':
    run()
