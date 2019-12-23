import torch
import torch.nn.functional as F


def look(vertices, viewpoints, direction=None, up=None):
    """
    "Look at" transformation of vertices.
    """
    assert (vertices.ndim == 3)

    if direction is None:
        direction = torch.as_tensor([0, 0, 1], dtype=torch.float32)
    if up is None:
        up = torch.as_tensor([0, 1, 0], dtype=torch.float32)

    if isinstance(viewpoints, list) or isinstance(viewpoints, tuple):
        viewpoints = torch.as_tensor(viewpoints, dtype=torch.float32)
    if viewpoints.ndim == 1:
        viewpoints = viewpoints[None, :]
    if direction.ndim == 1:
        direction = direction[None, :]
    if up.ndim == 1:
        up = up[None, :]

    # create new axes
    z_axis = F.normalize(direction)
    x_axis = F.normalize(torch.cross(up, z_axis))
    y_axis = F.normalize(torch.cross(z_axis, x_axis))

    # create rotation matrix: [bs, 3, 3]
    r = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), 1)
    if r.shape[0] != vertices.shape[0]:
        r = r.expand(vertices.shape)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != viewpoints.shape:
        viewpoints = viewpoints[:, None, :].expand(vertices.shape)
    vertices = vertices - viewpoints
    vertices = torch.matmul(vertices, r.transpose(1, 0))

    return vertices
