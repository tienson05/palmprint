"""
Khối Squeeze-and-Excitation (SE Block).

Module này thực hiện attention theo kênh bằng cách tái hiệu chỉnh
(recalibrate) các đặc trưng đầu vào. Đầu tiên, thông tin không gian
được nén lại thông qua Global Average Pooling (squeeze), sau đó
các phụ thuộc phi tuyến giữa các kênh được học bằng một mạng MLP
bottleneck (excitation). Các trọng số thu được sẽ được dùng để
tái tỉ lệ (scale) feature map theo từng kênh.

Ý tưởng chính:
    Nhấn mạnh các kênh quan trọng và giảm ảnh hưởng của các kênh nhiễu.

Tham số:
    channels (int): Số kênh đầu vào.
    reduction (int, optional): Tỉ lệ giảm chiều trong bottleneck. Mặc định: 16.

Input:
    Tensor kích thước (B, C, H, W)

Output:
    Tensor kích thước (B, C, H, W)
"""

import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1) # mỗi channel còn 1 số: (B, C, 1, 1)

        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), # giảm params
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False), # tăng params
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.squeeze(x).view(b, c) # 4D -> 2D: (B, C)
        y = self.excitation(y).view(b, c, 1, 1) # 2D -> 4D để nhân phía dưới

        return x * y