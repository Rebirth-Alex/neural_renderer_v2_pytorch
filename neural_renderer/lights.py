import torch


class Light:
    def __init__(self, color):
        self.color = color

    def to(self, device):
        self.color = self.color.to(device)


class DirectionalLight(Light):
    def __init__(self, color, direction, backside=False):
        super().__init__(color)
        self.direction = direction
        self.backside = backside

    def to(self, device):
        super().to(device)
        self.direction = self.direction.to(device)


class AmbientLight(Light):
    def __init__(self, color):
        super().__init__(color)


class SpecularLight(Light):
    def __init__(self, color, alpha=None, backside=False):
        super().__init__(color)
        self.backside = backside
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = torch.ones(color.shape[0], dtype=torch.float32)

    def to(self, device):
        super().to(device)
        self.alpha = self.alpha.to(device)
