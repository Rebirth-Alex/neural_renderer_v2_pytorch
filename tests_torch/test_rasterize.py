import os
import unittest

import chainer
import cupy as cp
import numpy as np
import pylab
import torch
from torch.nn.parameter import Parameter

import neural_renderer_torch
import neural_renderer_chainer
from neural_renderer_torch.rasterize_param import RasterizeParam, RasterizeHyperparam


class TestRasterize(unittest.TestCase):
    def test_forward_case1(self):
        # Render Chainer version
        target_num = 2

        vertices, faces = neural_renderer_chainer.load_obj('./tests_torch/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer_chainer.to_gpu((vertices_batch, faces))
        renderer = neural_renderer_chainer.Renderer()
        renderer.anti_aliasing = False
        renderer.viewpoints = neural_renderer_chainer.get_points_from_angles(2.732, 0, 0)
        images_chainer = renderer.render_silhouettes(vertices, faces).data[target_num]

        target_num = 2
        vertices, faces = neural_renderer_torch.load_obj('./tests_torch/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer_torch.to_gpu((vertices_batch, faces))

        renderer = neural_renderer_torch.Renderer()
        renderer.anti_aliasing = False
        renderer.viewpoints = neural_renderer_torch.get_points_from_angles(2.732, 0, 0)
        images_pytorch = renderer.render_silhouettes(vertices, faces).data[target_num]

        np.testing.assert_allclose(images_pytorch.cpu().numpy(), images_chainer.get(), atol=2e-3)

    def test_forward_case2(self):
        data = [
            [
                './tests_torch/data/4e49873292196f02574b5684eaec43e9/model.obj',
                neural_renderer_torch.get_points_from_angles(2.5, 10, -90),
                './tests_torch/data/4e49873292196f02574b5684eaec43e9.png',
            ],
            [
                './tests_torch/data/1cde62b063e14777c9152a706245d48/model.obj',
                neural_renderer_torch.get_points_from_angles(2.5, 10, 60),
                './tests_torch/data/1cde62b063e14777c9152a706245d48.png',
            ]
        ]

        for i, (filename, viewpoint, reference) in enumerate(data):
            # Render Chainer
            renderer = neural_renderer_chainer.Renderer()
            renderer.draw_backside = False
            renderer.viewpoints = viewpoint
            vertices, faces, vertices_t, faces_t, textures = neural_renderer_chainer.load_obj(filename,
                                                                                              load_textures=True)
            vertices, faces, vertices_t, faces_t, textures = neural_renderer_chainer.to_gpu(
                (vertices[None, :, :], faces, vertices_t[None, :, :], faces_t, textures[None, :, :, :]))
            images = renderer.render(vertices, faces, vertices_t, faces_t, textures).data
            image_chainer = images[0].transpose((1, 2, 0)).get()

            # Render PyTorch
            renderer = neural_renderer_torch.Renderer()
            renderer.draw_backside = False
            renderer.viewpoints = viewpoint
            vertices, faces, vertices_t, faces_t, textures = neural_renderer_torch.load_obj(filename, load_textures=True)
            vertices, faces, vertices_t, faces_t, textures = neural_renderer_torch.to_gpu(
                (vertices[None, :, :], faces, vertices_t[None, :, :], faces_t, textures[None, :, :, :]))

            images = renderer.render(vertices, faces, vertices_t, faces_t, textures)
            images = images.cpu().numpy()
            image_pytorch = images[0].transpose((1, 2, 0))

            assert np.mean(np.abs(image_chainer - image_pytorch)) < 1e-4

    def test_forward_case3(self):
        # Render with Chainer
        target_num = 2
        vertices, faces = neural_renderer_chainer.load_obj('./tests_torch/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer_chainer.to_gpu((vertices_batch, faces))

        renderer = neural_renderer_chainer.Renderer()
        renderer.anti_aliasing = False
        renderer.viewpoints = neural_renderer_chainer.get_points_from_angles(2, 30., 0)
        images = renderer.render_depth(vertices, faces).data[target_num]
        images_chainer = (images - images.min()) / (images.max() - images.min())
        images_chainer = images_chainer.get()

        # Render with PyTorch
        target_num = 2
        vertices, faces = neural_renderer_torch.load_obj('./tests_torch/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer_torch.to_gpu((vertices_batch, faces))

        renderer = neural_renderer_torch.Renderer()
        renderer.anti_aliasing = False
        renderer.viewpoints = neural_renderer_torch.get_points_from_angles(2, 30., 0)
        images = renderer.render_depth(vertices, faces).data[target_num]
        images_pytorch = (images - images.min()) / (images.max() - images.min())
        images_pytorch = images_pytorch.cpu().numpy()

        assert np.mean(np.abs(images_pytorch - images_chainer)) < 1e-4

    def test_forward_case4(self):
        # Render with Chainer

        target_num = 2
        vertices, faces = neural_renderer_chainer.load_obj('./tests_torch/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer_chainer.to_gpu((vertices_batch, faces))
        vertices_t, faces_t, textures = neural_renderer_chainer.create_textures(faces.shape[0])
        vertices_t = chainer.functions.tile(vertices_t[None, :, :], (4, 1, 1)).data
        textures = chainer.functions.tile(textures[None, :, :, :], (4, 1, 1, 1)).data
        vertices_t = chainer.cuda.to_gpu(vertices_t)
        faces_t = chainer.cuda.to_gpu(faces_t)
        textures = chainer.cuda.to_gpu(textures)

        lights = []
        light_color = cp.asarray([[0.47481096, 0.7131511, 0.4510043],
                                  [0.49120015, 0.161955, 0.71638113],
                                  [0.32655084, 0.7805874, 0.7682426],
                                  [0.42193118, 0.90416473, 0.5267034]]).astype(cp.float32)
        light_direction = cp.asarray([[0.328245, 0.8916046, 0.31189483],
                                      [0.99824226, 0.05838178, 0.00867782],
                                      [0.35747865, 0.61983925, 0.6985467],
                                      [0.0393897, 0.6937492, 0.7191179]]).astype(cp.float32)
        lights.append(neural_renderer_chainer.DirectionalLight(light_color, light_direction))

        light_color = cp.asarray([[0.2732121, 0.09439224, 0.38380036],
                                  [0.06487979, 0.02794903, 0.261018],
                                  [0.28739947, 0.2996951, 0.42412606],
                                  [0.10019363, 0.26517034, 0.07372955]]).astype(cp.float32)
        lights.append(neural_renderer_chainer.AmbientLight(light_color))

        light_color = cp.asarray([[0.32410273, 0.24369295, 0.3126097],
                                  [0.3456873, 0.24514836, 0.21663068],
                                  [0.33004418, 0.25533527, 0.48039845],
                                  [0.29468802, 0.44377372, 0.10724097]]).astype(cp.float32)
        lights.append(neural_renderer_chainer.SpecularLight(light_color))

        renderer = neural_renderer_chainer.Renderer()
        renderer.viewpoints = neural_renderer_chainer.get_points_from_angles(2.732, 30, 30)
        renderer.draw_backside = False
        images_chainer = renderer.render_rgb(vertices, faces, vertices_t, faces_t, textures, lights=lights).data[
            target_num]
        images_chainer = images_chainer.get()

        # Render with PyTorch
        target_num = 2
        vertices, faces = neural_renderer_torch.load_obj('./tests_torch/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer_torch.to_gpu((vertices_batch, faces))
        vertices_t, faces_t, textures = neural_renderer_torch.create_textures(faces.shape[0])
        vertices_t = torch.as_tensor(vertices_t[None, :, :]).expand((4, *vertices_t.shape))
        faces_t = torch.as_tensor(faces_t)
        textures = torch.as_tensor(textures[None, :, :, :]).expand((4, *textures.shape))
        vertices_t = vertices_t.cuda()
        faces_t = faces_t.cuda()
        textures = textures.cuda()

        lights = []

        light_color = torch.as_tensor([[0.47481096, 0.7131511, 0.4510043],
                                       [0.49120015, 0.161955, 0.71638113],
                                       [0.32655084, 0.7805874, 0.7682426],
                                       [0.42193118, 0.90416473, 0.5267034]]).type(torch.float32)
        light_direction = torch.as_tensor([[0.328245, 0.8916046, 0.31189483],
                                           [0.99824226, 0.05838178, 0.00867782],
                                           [0.35747865, 0.61983925, 0.6985467],
                                           [0.0393897, 0.6937492, 0.7191179]]).type(torch.float32)
        lights.append(neural_renderer_torch.DirectionalLight(light_color, light_direction))

        light_color = torch.as_tensor([[0.2732121, 0.09439224, 0.38380036],
                                       [0.06487979, 0.02794903, 0.261018],
                                       [0.28739947, 0.2996951, 0.42412606],
                                       [0.10019363, 0.26517034, 0.07372955]]).type(torch.float32)
        lights.append(neural_renderer_torch.AmbientLight(light_color))

        light_color = torch.as_tensor([[0.32410273, 0.24369295, 0.3126097],
                                       [0.3456873, 0.24514836, 0.21663068],
                                       [0.33004418, 0.25533527, 0.48039845],
                                       [0.29468802, 0.44377372, 0.10724097]]).type(torch.float32)
        lights.append(neural_renderer_torch.SpecularLight(light_color))

        renderer = neural_renderer_torch.Renderer()
        renderer.viewpoints = neural_renderer_torch.get_points_from_angles(2.732, 30, 30)
        renderer.draw_backside = False
        images = renderer.render_rgb(vertices, faces, vertices_t, faces_t, textures, lights=lights).data[target_num]
        images_pytorch = images.cpu().numpy()

        assert np.mean(np.abs(images_pytorch - images_chainer)) < 1e-4

    def test_backward_case1(self):
        vertices = [
            [0.1, 0.1, 1.],
            [-0.1, 0.1, 1.],
            [-0.1, -0.1, 1.],
            [0.1, -0.1, 1.],
        ]
        faces = [[0, 1, 2], [0, 2, 3]]

        ref = neural_renderer_torch.imread('./tests_torch/data/gradient.png')
        ref = 1 - ref
        ref = ref[:, :, 0]
        ref = torch.as_tensor(ref).cuda()

        vertices = np.array(vertices, 'float32')
        faces = np.array(faces, 'int32')
        vertices, faces = neural_renderer_torch.to_gpu((vertices, faces))
        vertices = Parameter(vertices, True)
        faces = torch.as_tensor(faces)

        optimizer = torch.optim.Adam([vertices], lr=0.005)
        ref = torch.as_tensor(ref)

        os.makedirs("tmp", exist_ok=True)

        for i in range(350):
            params = RasterizeParam()
            hyperparams = RasterizeHyperparam(image_size=256, anti_aliasing=False)
            images = neural_renderer_torch.rasterize_silhouettes(vertices[None, :, :], faces, params, hyperparams)

            image = images[0]
            iou = torch.sum(image * ref) / torch.sum(image + ref - image * ref)
            iou = torch.as_tensor(1, dtype=torch.float32, device=iou.device) - iou
            loss = iou

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pylab.imsave('tmp/t%d.png' % i, image.data.cpu().numpy())
            print(i, loss.data, iou.data)

            if float(iou.data) < 0.01:
                return
        raise Exception


if __name__ == '__main__':
    unittest.main()
