import torch
from torch.autograd import Function


def pad_zeros(x, size, axis, side='both'):
    if axis == 1:
        pad = torch.zeros((x.shape[0], size, x.shape[2], x.shape[3]), dtype=torch.float32)
    elif axis == 2:
        pad = torch.zeros((x.shape[0], x.shape[1], size, x.shape[3]), dtype=torch.float32)
    elif axis == 3:
        pad = torch.zeros((x.shape[0], x.shape[1], x.shape[2], size), dtype=torch.float32)
    if side == 'both':
        x = torch.cat((pad, x, pad), axis)
    elif side == 'left':
        x = torch.cat((pad, x), axis)
    elif side == 'right':
        x = torch.cat((x, pad), axis)
    return x


def maximum(data_right, data_left, eps=1e-4):
    max_map = torch.max(data_left, data_right)
    min_map = torch.abs(data_right - data_left) < eps
    rl_map = data_right > data_left
    else_map = not (max_map | min_map | rl_map)

    data3 = torch.zeros_like(data_right)
    data3[max_map] = 0
    data3[min_map] = 0
    data3[rl_map] = -data_right[rl_map]
    data3[else_map] = data_left[else_map]

    return data3


class Differentiation(Function):

    def forward(self, images, coordinates):
        return images

    def backward(self, inputs, gradients):
        images, coordinates = inputs
        grad_output = gradients[0]
        batch_size, image_size, _, num_channels = images.shape
        step = 2. / image_size

        grad_images = gradients[0].copy()

        grad_y_r = -((images[:, :-1, :] - images[:, 1:, :]) * grad_output[:, 1:, :]).sum(-1) / step
        grad_y_r = pad_zeros(grad_y_r[:, :, :, None], 1, 1, 'right') + pad_zeros(grad_y_r[:, :, :, None], 1, 1, 'left')
        grad_y_l = -((images[:, 1:, :] - images[:, :-1, :]) * grad_output[:, :-1, :]).sum(-1) / step
        grad_y_l = pad_zeros(grad_y_l[:, :, :, None], 1, 1, 'left') + pad_zeros(grad_y_l[:, :, :, None], 1, 1, 'right')
        grad_y = maximum(grad_y_r, grad_y_l)

        grad_x_r = -((images[:, :, :-1] - images[:, :, 1:]) * grad_output[:, :, 1:]).sum(-1) / step
        grad_x_r = pad_zeros(grad_x_r[:, :, :, None], 1, 2, 'right') + pad_zeros(grad_x_r[:, :, :, None], 1, 2, 'left')
        grad_x_l = -((images[:, :, 1:] - images[:, :, :-1]) * grad_output[:, :, :-1]).sum(-1) / step
        grad_x_l = pad_zeros(grad_x_l[:, :, :, None], 1, 2, 'left') + pad_zeros(grad_x_l[:, :, :, None], 1, 2, 'right')
        grad_x = maximum(grad_x_r, grad_x_l)

        grad_loss_xy = torch.cat((grad_x, grad_y), -1)

        return grad_images, grad_loss_xy


def differentiation(images, coordinates):
    return Differentiation()(images, coordinates)
