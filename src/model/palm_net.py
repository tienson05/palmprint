"""
PalmNet - Mạng CNN sâu cho trích xuất đặc trưng lòng bàn tay (palmprint).

Kiến trúc này được xây dựng dựa trên ResNet, kết hợp với các khối
Squeeze-and-Excitation (SE) nhằm tăng cường khả năng mô hình hóa
phụ thuộc giữa các kênh (channel-wise attention).

Mạng bao gồm:
    - Stem: Lớp convolution ban đầu (7x7) + BatchNorm + ReLU + MaxPool
      để giảm nhanh kích thước không gian và trích xuất đặc trưng cơ bản.
    - Các tầng residual (layer1 → layer4): Gồm các ResBlock tích hợp SE,
      giúp học đặc trưng sâu hơn, đồng thời giảm dần kích thước không gian
      và tăng số lượng kênh (64 → 128 → 256 → 512).
    - Head: Global Average Pooling + Fully Connected để chuyển feature map
      thành vector embedding 128 chiều.

Đầu ra:
    Vector embedding được chuẩn hóa L2 (unit norm)

Ý tưởng chính:
    - Học biểu diễn đặc trưng mạnh thông qua residual learning
    - Tăng cường kênh quan trọng bằng SE attention
    - Chuẩn hóa embedding để so sánh bằng cosine similarity

Input:
    Tensor kích thước (B, 1, 224, 224)

Output:
    Tensor kích thước (B, 128), đã được chuẩn hóa L2
"""

from torch import nn
import torch.nn.functional as F
from src.model.res_block import ResBlock

class PalmNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 224x224
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), # 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 56
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2), # 28
            ResBlock(128, 128),
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2), # 14
            ResBlock(256, 256),
        )

        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2), # 7
            ResBlock(512, 512),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.head(out)
        out = F.normalize(out, p=2, dim=1)
        return out