"""
Residual Block tích hợp Squeeze-and-Excitation (SE).

Module này triển khai một khối residual theo kiến trúc ResNet, kết hợp
với cơ chế attention theo kênh (SE Block) nhằm tăng khả năng biểu diễn
của mô hình. Nhánh chính gồm hai lớp convolution 3x3 kèm BatchNorm và ReLU,
sau đó được tái hiệu chỉnh bằng SEBlock. Nhánh shortcut được sử dụng để
truyền trực tiếp thông tin đầu vào, giúp cải thiện gradient flow.

Nếu kích thước (spatial hoặc channel) giữa input và output không khớp,
shortcut sẽ được điều chỉnh bằng convolution 1x1 (projection) kèm BatchNorm
để đảm bảo có thể thực hiện phép cộng residual.

Ý tưởng chính:
    Học phần dư (residual) của đặc trưng đầu vào, đồng thời nhấn mạnh
    các kênh quan trọng thông qua SEBlock.

Tham số:
    in_channels (int): Số kênh đầu vào.
    out_channels (int): Số kênh đầu ra.
    stride (int, optional): Bước stride của convolution đầu tiên. Mặc định: 1.

Input:
    Tensor kích thước (B, C, H, W)

Output:
    Tensor kích thước (B, out_channels, H', W')
"""

import torch.nn as nn
from src.model.se_block import SEBlock

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            # Điều chỉnh shortcut để khớp shape trước khi cộng
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # Giống shape không cần giảm
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += identity
        out = self.relu(out)

        return out