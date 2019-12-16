import unittest

import numpy as np
import torch

import neural_renderer


class TestPerspective(unittest.TestCase):
    def test_case1(self):
        v_in = [1, 2, 10]
        v_out = [np.sqrt(3) / 10, 2 * np.sqrt(3) / 10, 10]
        vertices = np.array(v_in, 'float32')
        vertices = vertices[None, None, :]
        transformed = neural_renderer.perspective(torch.as_tensor(vertices))
        np.testing.assert_allclose(transformed.flatten().numpy(), np.asarray(v_out, np.float32), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
