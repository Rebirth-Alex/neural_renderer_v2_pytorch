import torch

from .utils import pad_zeros, maximum


class Differentiation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, images, coordinates):
        ctx.save_for_backward(images)
        return images

    @staticmethod
    def backward(ctx, gradients):
        images, = ctx.saved_tensors
        grad_output = gradients
        batch_size, image_size, _, num_channels = images.shape
        step = 2. / image_size

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

        # faces: [bs, nf, 3, 3]
        # face_index_map: [bs, is, is]
        # weight_map: [bs, is, is, 3]
        return gradients, grad_loss_xy


def differentiation(images, coordinates):
    return Differentiation().apply(images, coordinates)


class ComputeSilhouette(torch.autograd.Function):
    @staticmethod
    def forward(ctx, face_index_map):
        silhouettes = (0 <= face_index_map).type(torch.float32)[:, :, :, None]
        return silhouettes,

    @staticmethod
    def backward(ctx, inputs, gradients):
        return gradients


def compute_silhouette(face_index_map):
    grad_buf = face_index_map.requires_grad
    face_index_map.requires_grad = True
    res = ComputeSilhouette().apply(face_index_map)
