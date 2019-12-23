import glob
import os
import unittest

import numpy

import neural_renderer_torch


class TestSaveObj(unittest.TestCase):
    def test_case1(self):
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
        filename_tmp = './tests_torch/data/tmp.obj'

        renderer = neural_renderer_torch.Renderer()
        renderer.draw_backside = False
        for i, (filename, viewpoint, reference) in enumerate(data):
            renderer.viewpoints = viewpoint
            ref = neural_renderer_torch.imread(reference)

            vertices, faces, vertices_t, faces_t, textures = neural_renderer_torch.load_obj(filename,
                                                                                            load_textures=True)
            neural_renderer_torch.save_obj(filename_tmp, vertices, faces, vertices_t, faces_t, textures)
            vertices, faces, vertices_t, faces_t, textures = neural_renderer_torch.load_obj(filename_tmp,
                                                                                            load_textures=True)
            vertices, faces, vertices_t, faces_t, textures = neural_renderer_torch.to_gpu(
                (vertices[None, :, :], faces, vertices_t[None, :, :], faces_t, textures[None, :, :, :]))

            images = renderer.render(vertices, faces, vertices_t, faces_t, textures).data
            image = images[0].transpose((1, 2, 0))

            numpy.testing.assert_allclose(ref, image, atol=1e-2, rtol=1e-2)

        for f in glob.glob('./tests_torch/data/tmp*'):
            os.remove(f)


if __name__ == '__main__':
    unittest.main()
