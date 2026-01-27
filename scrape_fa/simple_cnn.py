import torch
import torch.nn as nn

class SimpleCNN(nn.Module):

    ALLOWED_KWARGS = {"kernel_size"}

    def __init__(self, n_channels, n_classes=1, **kwargs):
        super().__init__()

        unexpected_keys = set(kwargs) - self.ALLOWED_KWARGS
        if unexpected_keys:
            raise TypeError(
                f"Unexpected keyword argument(s) for {self.__class__.__name__}: "
                f"{', '.join(sorted(unexpected_keys))}"
            )

        kernel_size = kwargs.pop("kernel_size", 3)

        self.config = {
            "n_channels": n_channels,
            "n_classes": n_classes,
            "kernel_size": kernel_size,
        }

        self.conv1 = nn.Conv2d(n_channels, 8, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=4, dilation=4)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(8, 8, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(8, 8, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu6 = nn.ReLU(inplace=True)
        self.convend = nn.Conv2d(8, n_classes, kernel_size=1)  # output heatmap

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.convend(x)
        return x

    @classmethod
    def from_pth(cls, file_name: str, map_location=None, model_args=None):
        checkpoint = torch.load(file_name, map_location=map_location)

        model_args = model_args or {}
        if isinstance(checkpoint, dict) and 'model_args' in checkpoint:
            model_args = model_args | checkpoint["model_args"]

        model = cls(**model_args)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        return model

