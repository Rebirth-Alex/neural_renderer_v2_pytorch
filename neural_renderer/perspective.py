import torch


def perspective(vertices, angle=30.):
    assert (vertices.ndim == 3)
    device = vertices.device
    if isinstance(angle, float) or isinstance(angle, int):
        angle = torch.as_tensor(angle, dtype=torch.float32, device=device)
    angle = angle / 180. * 3.1416
    angle = angle[None].expand((vertices.shape[0],))

    width = torch.tan(angle)
    width = width[:, None].expand(vertices.shape[:2])
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = torch.cat((x[:, :, None], y[:, :, None], z[:, :, None]), 2)
    return vertices
