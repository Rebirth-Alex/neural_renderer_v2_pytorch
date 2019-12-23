import unittest

import chainer
import chainer.functions as cf
import chainer.testing
import cupy as cp
import imageio
import numpy as np

import neural_renderer_chainer


class Parameter(chainer.Link):
    def __init__(self, x):
        super(Parameter, self).__init__()
        with self.init_scope():
            self.x = chainer.Parameter(x)

    def __call__(self):
        return self.x


class TestRasterize(unittest.TestCase):
    def stest_forward_case1(self):
        # load reference image by blender
        ref = imageio.imread('./tests_chainer/data/teapot_blender.png')
        ref = (ref.min(-1) != 255).astype('float32')
        ref = chainer.cuda.to_gpu(ref)

        target_num = 2
        vertices, faces = neural_renderer_chainer.load_obj('./tests_chainer/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer_chainer.to_gpu((vertices_batch, faces))

        renderer = neural_renderer_chainer.Renderer()
        renderer.anti_aliasing = False
        renderer.viewpoints = neural_renderer_chainer.get_points_from_angles(2.732, 0, 0)
        images = renderer.render_silhouettes(vertices, faces).data[target_num]

        chainer.testing.assert_allclose(images, ref, atol=2e-3)

    def test_forward_case2(self):
        data = [
            [
                './tests_chainer/data/4e49873292196f02574b5684eaec43e9/model.obj',
                neural_renderer_chainer.get_points_from_angles(2.5, 10, -90),
                './tests_chainer/data/4e49873292196f02574b5684eaec43e9.png',
            ],
            [
                './tests_chainer/data/1cde62b063e14777c9152a706245d48/model.obj',
                neural_renderer_chainer.get_points_from_angles(2.5, 10, 60),
                './tests_chainer/data/1cde62b063e14777c9152a706245d48.png',
            ]
        ]

        renderer = neural_renderer_chainer.Renderer()
        renderer.draw_backside = False
        for i, (filename, viewpoint, reference) in enumerate(data):
            renderer.viewpoints = viewpoint
            ref = neural_renderer_chainer.imread(reference)

            vertices, faces, vertices_t, faces_t, textures = neural_renderer_chainer.load_obj(filename,
                                                                                              load_textures=True)
            vertices, faces, vertices_t, faces_t, textures = neural_renderer_chainer.to_gpu(
                (vertices[None, :, :], faces, vertices_t[None, :, :], faces_t, textures[None, :, :, :]))

            images = renderer.render(vertices, faces, vertices_t, faces_t, textures).data
            image = images[0].transpose((1, 2, 0))
            # imageio.toimage(image.get(), cmin=0, cmax=1).save(reference)

            chainer.testing.assert_allclose(ref, image, atol=1e-2)

    def stest_forward_case3(self):
        # load reference image by blender
        ref = imageio.imread('./tests_chainer/data/teapot_depth.png')
        ref = ref.astype('float32') / 255.
        ref = chainer.cuda.to_gpu(ref)

        target_num = 2
        vertices, faces = neural_renderer_chainer.load_obj('./tests_chainer/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer_chainer.to_gpu((vertices_batch, faces))

        renderer = neural_renderer_chainer.Renderer()
        renderer.anti_aliasing = False
        renderer.viewpoints = neural_renderer_chainer.get_points_from_angles(2, 30., 0)
        images = renderer.render_depth(vertices, faces).data[target_num]
        images = (images - images.min()) / (images.max() - images.min())

        chainer.testing.assert_allclose(images, ref, atol=2e-3)

    def stest_forward_case4(self):
        # lights
        ref = imageio.imread('./tests_chainer/data/teapot_blender.png')
        ref = (ref.min(-1) != 255).astype('float32')
        ref = chainer.cuda.to_gpu(ref)

        target_num = 2
        vertices, faces = neural_renderer_chainer.load_obj('./tests_chainer/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer_chainer.to_gpu((vertices_batch, faces))
        vertices_t, faces_t, textures = neural_renderer_chainer.create_textures(faces.shape[0])
        vertices_t = cf.tile(vertices_t[None, :, :], (4, 1, 1)).data
        textures = cf.tile(textures[None, :, :, :], (4, 1, 1, 1)).data
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
        images = renderer.render_rgb(vertices, faces, vertices_t, faces_t, textures, lights=lights).data[target_num]

        import pylab
        # pylab.imshow(images.get().transpose((1, 2, 0))[:, :, :3])
        # pylab.show()
        pylab.imsave("test_rasterize_case4.jpg", images.get().transpose((1, 2, 0)) / images.get().max())

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

        ref = neural_renderer_chainer.imread('./tests_chainer/data/gradient.png')
        ref = 1 - ref
        ref = ref[:, :, 0]
        ref = chainer.cuda.to_gpu(ref)

        vertices = np.array(vertices, 'float32')
        faces = np.array(faces, 'int32')
        vertices, faces, ref = neural_renderer_chainer.to_gpu((vertices, faces, ref))
        vertices = Parameter(vertices)
        optimizer = chainer.optimizers.Adam(0.003, beta1=0.5)
        optimizer.setup(vertices)

        for i in range(350):
            images = neural_renderer_chainer.rasterize_silhouettes(
                vertices()[None, :, :], faces, image_size=256, anti_aliasing=False)
            image = images[0]

            iou = cf.sum(image * ref) / cf.sum(image + ref - image * ref)
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
