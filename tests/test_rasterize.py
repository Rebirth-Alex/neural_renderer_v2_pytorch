import unittest

import imageio
import numpy as np
import torch
import torch.nn.functional as F

import neural_renderer


class TestRasterize(unittest.TestCase):
    def stest_forward_case1(self):
        # load reference image by blender
        ref = imageio.imread('./tests/data/teapot_blender.png')
        ref = (ref.min(-1) != 255).astype('float32')
        ref = torch.as_tensor(ref).cuda()

        target_num = 2
        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer.to_gpu((vertices_batch, faces))

        renderer = neural_renderer.Renderer()
        renderer.anti_aliasing = False
        renderer.viewpoints = neural_renderer.get_points_from_angles(2.732, 0, 0)
        images = renderer.render_silhouettes(vertices, faces).data[target_num]

        np.testing.assert_allclose(images, ref, atol=2e-3)

    def stest_forward_case2(self):
        data = [
            [
                './tests/data/4e49873292196f02574b5684eaec43e9/model.obj',
                neural_renderer.get_points_from_angles(2.5, 10, -90),
                './tests/data/4e49873292196f02574b5684eaec43e9.png',
            ],
            [
                './tests/data/1cde62b063e14777c9152a706245d48/model.obj',
                neural_renderer.get_points_from_angles(2.5, 10, 60),
                './tests/data/1cde62b063e14777c9152a706245d48.png',
            ]
        ]

        renderer = neural_renderer.Renderer()
        renderer.draw_backside = False
        for i, (filename, viewpoint, reference) in enumerate(data):
            renderer.viewpoints = viewpoint
            ref = neural_renderer.imread(reference)

            vertices, faces, vertices_t, faces_t, textures = neural_renderer.load_obj(filename, load_textures=True)
            vertices, faces, vertices_t, faces_t, textures = neural_renderer.to_gpu(
                (vertices[None, :, :], faces, vertices_t[None, :, :], faces_t, textures[None, :, :, :]))

            images = renderer.render(vertices, faces, vertices_t, faces_t, textures).data
            image = images[0].transpose((1, 2, 0))

            np.testing.assert_allclose(ref, image, atol=1e-2)

    def stest_forward_case3(self):
        # load reference image by blender
        ref = imageio.imread('./tests/data/teapot_depth.png')
        ref = ref.astype('float32') / 255.
        ref = torch.as_tensor(ref).cuda()

        target_num = 2
        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer.to_gpu((vertices_batch, faces))

        renderer = neural_renderer.Renderer()
        renderer.anti_aliasing = False
        renderer.viewpoints = neural_renderer.get_points_from_angles(2, 30., 0)
        images = renderer.render_depth(vertices, faces).data[target_num]
        images = (images - images.min()) / (images.max() - images.min())
        # imageio.toimage(images.get()).save('./tests/data/teapot_depth.png')

        np.testing.assert_allclose(images, ref, atol=2e-3)

    def test_forward_case4(self):
        # lights
        ref = imageio.imread('./tests/data/teapot_blender.png')
        ref = (ref.min(-1) != 255).astype('float32')
        ref = torch.as_tensor(ref).cuda()

        target_num = 2
        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer.to_gpu((vertices_batch, faces))
        vertices_t, faces_t, textures = neural_renderer.create_textures(faces.shape[0])
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
        lights.append(neural_renderer.DirectionalLight(light_color, light_direction))

        light_color = torch.as_tensor([[0.2732121, 0.09439224, 0.38380036],
                                       [0.06487979, 0.02794903, 0.261018],
                                       [0.28739947, 0.2996951, 0.42412606],
                                       [0.10019363, 0.26517034, 0.07372955]]).type(torch.float32)
        lights.append(neural_renderer.AmbientLight(light_color))

        light_color = torch.as_tensor([[0.32410273, 0.24369295, 0.3126097],
                                       [0.3456873, 0.24514836, 0.21663068],
                                       [0.33004418, 0.25533527, 0.48039845],
                                       [0.29468802, 0.44377372, 0.10724097]]).type(torch.float32)
        lights.append(neural_renderer.SpecularLight(light_color))

        renderer = neural_renderer.Renderer()
        renderer.viewpoints = neural_renderer.get_points_from_angles(2.732, 30, 30)
        renderer.draw_backside = False
        images = renderer.render_rgb(vertices, faces, vertices_t, faces_t, textures, lights=lights).data[target_num]
        images = images.cpu().numpy().transpose((1, 2, 0))

        import pylab
        pylab.imshow(images.cpu().numpy().transpose((1, 2, 0))[:, :, :3])
        pylab.show()

    def stest_backward_case1(self):
        vertices = [
            [0.1, 0.1, 1.],
            [-0.1, 0.1, 1.],
            [-0.1, -0.1, 1.],
            [0.1, -0.1, 1.],
        ]
        faces = [[0, 1, 2], [0, 2, 3]]
        # vertices = [
        #     [0.3, 0.4, 1.],
        #     [-0.3, 0.6, 1.],
        #     [-0.3, 0.62, 1.],
        # ]
        # faces = [[0, 1, 2]]

        ref = neural_renderer.imread('./tests/data/gradient.png')
        ref = 1 - ref
        ref = ref[:, :, 0]
        ref = torch.as_tensor(ref).cuda()

        vertices = np.array(vertices, 'float32')
        faces = np.array(faces, 'int32')
        vertices, faces, ref = neural_renderer.to_gpu((vertices, faces, ref))
        vertices = torch.as_tensor(vertices)
        optimizer = torch.optim.Adam(0.003)
        optimizer.setup(vertices)

        for i in range(350):
            images = neural_renderer.rasterize_silhouettes(
                vertices()[None, :, :], faces, image_size=256, anti_aliasing=False)
            image = images[0]

            iou = torch.sum(image * ref) / torch.sum(image + ref - image * ref)
            iou = 1 - iou
            loss = iou

            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

            # imageio.toimage(image.data.get()).save('../tmp/t%d.png' % i)
            # print i, loss.data, iou.data

            if float(iou.data) < 0.01:
                return
        raise Exception


if __name__ == '__main__':
    unittest.main()
