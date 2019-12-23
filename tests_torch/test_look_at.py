import unittest

import numpy as np
import torch

import neural_renderer_torch


class TestLookAt(unittest.TestCase):
    def test_case1(self):
        viewpointss = [
            [1, 0, 1],
            [0, 0, -10],
            [-1, 1, 0],
        ]
        answers = [
            [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
            [1, 0, 10],
            [0, np.sqrt(2) / 2, 3. / 2. * np.sqrt(2)],
        ]
        vertices = np.array([1, 0, 0], 'float32')
        vertices = vertices[None, None, :]
        for e, a in zip(viewpointss, answers):
            viewpoints = np.array(e, 'float32')
            transformed = neural_renderer_torch.look_at(torch.as_tensor(vertices), torch.as_tensor(viewpoints))
            np.testing.assert_allclose(transformed.flatten().numpy(), np.array(a))


if __name__ == '__main__':
    unittest.main()
