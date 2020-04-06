import unittest

import numpy as np
import torch

import neural_renderer_torch


class TestDifferentiation(unittest.TestCase):
    def test_backward(self):
        images = np.random.normal(size=(10, 32, 32, 3)).astype('float32')

        x = np.tile(np.arange(32).astype('float32')[None, None, :, None], (10, 32, 1, 1))
        y = np.tile(np.arange(32).astype('float32')[None, :, None, None], (10, 1, 32, 1))

        coordinates = np.concatenate((x, y), axis=-1)
        coordinates = ((coordinates / 31) * 2 - 1) * 31. / 32.

        noise = np.random.normal(size=(10, 32, 32, 3)).astype('float32')
        step = 2 / 32.

        device = torch.device('cuda:0')
        images = torch.tensor(images, device=device)
        coordinates = torch.tensor(coordinates, device=device, requires_grad=True)
        noise = torch.tensor(noise, device=device)

        loss = torch.sum(neural_renderer_torch.differentiation(images, coordinates) * noise)
        loss.backward()

        grad_coordinates = coordinates.grad

        for i in range(100):
            yi = np.random.randint(1, 31)
            xi = np.random.randint(1, 31)

            images_yb = images.clone()
            images_yb[:, yi - 1, xi] = images[:, yi, xi].clone()
            images_yb[:, yi, xi] = images[:, yi + 1, xi].clone()
            grad_yb = ((images_yb - images) * noise).sum((1, 2, 3)) / step
            grad_yb = torch.min(grad_yb, torch.zeros_like(grad_yb))

            images_yt = images.clone()
            images_yt[:, yi + 1, xi] = images[:, yi, xi].clone()
            images_yt[:, yi, xi] = images[:, yi - 1, xi].clone()
            grad_yt = ((images_yt - images) * noise).sum((1, 2, 3)) / step
            grad_yt = torch.min(grad_yt, torch.zeros_like(grad_yt))

            grad_y_abs = torch.max(torch.abs(grad_yb), torch.abs(grad_yt))

            assert torch.allclose(grad_y_abs, torch.abs(grad_coordinates[:, yi, xi, 1]), rtol=1e-04, atol=0)

            images_xl = images.clone()
            images_xl[:, yi, xi - 1] = images[:, yi, xi].clone()
            images_xl[:, yi, xi] = images[:, yi, xi + 1].clone()
            grad_xl = ((images_xl - images) * noise).sum((1, 2, 3)) / step
            grad_xl = torch.min(grad_xl, torch.zeros_like(grad_xl))

            images_xr = images.clone()
            images_xr[:, yi, xi + 1] = images[:, yi, xi].clone()
            images_xr[:, yi, xi] = images[:, yi, xi - 1].clone()
            grad_xr = ((images_xr - images) * noise).sum((1, 2, 3)) / step
            grad_xr = torch.min(grad_xr, torch.zeros_like(grad_xr))

            grad_x_abs = torch.max(torch.abs(grad_xl), torch.abs(grad_xr))

            assert torch.allclose(grad_x_abs, torch.abs(grad_coordinates[:, yi, xi, 0]), rtol=1e-04, atol=0)


if __name__ == '__main__':
    unittest.main()
