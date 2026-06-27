import torch
import torch.nn as nn


def _get_num_classes(dataset_info, num_classes):
    if num_classes is not None:
        return num_classes
    if dataset_info is None:
        raise ValueError("num_classes must be provided when dataset_info is None")
    return dataset_info["num_classes"]


def _init_with_seed(seed, init_fn):
    if seed is None:
        init_fn()
        return
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        init_fn()


class LeNet5(nn.Module):
    def __init__(self, num_classes, in_channels=1, use_batch_norm=False, seed=None):
        super().__init__()

        def _init():
            self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0)
            self.bn1 = nn.BatchNorm2d(6) if use_batch_norm else nn.Identity()
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
            self.bn2 = nn.BatchNorm2d(16) if use_batch_norm else nn.Identity()
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.bn3 = nn.BatchNorm1d(120) if use_batch_norm else nn.Identity()
            self.fc2 = nn.Linear(120, 84)
            self.bn4 = nn.BatchNorm1d(84) if use_batch_norm else nn.Identity()
            self.fc = nn.Linear(84, num_classes)
            self.relu = nn.ReLU(inplace=True)

        _init_with_seed(seed, _init)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 6, 24, 24)
        x = self.pool(x)  # (B, 6, 12, 12)
        x = self.relu(self.bn2(self.conv2(x)))  # (B, 16, 8, 8)
        x = self.pool(x)  # (B, 16, 4, 4)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.relu(self.bn3(self.fc1(x)))  # (B, 120)
        x = self.relu(self.bn4(self.fc2(x)))  # (B, 84)
        x = self.fc(x)  # (B, num_classes)
        return x


def mnist_lenet5(dataset_info, in_channels=1, num_classes=None, use_batch_norm=True, seed=None, **kwargs):
    num_classes = _get_num_classes(dataset_info, num_classes)
    return LeNet5(
        num_classes=num_classes,
        in_channels=in_channels,
        use_batch_norm=use_batch_norm,
        seed=seed,
    )
