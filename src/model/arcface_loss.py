"""
ArcFace Loss (Additive Angular Margin Loss)

Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
       https://arxiv.org/abs/1801.07698

Ý tưởng:
    Thay vì học khoảng cách Euclidean (Triplet Loss), ArcFace tối ưu góc giữa
    embedding và weight vector của từng class. Margin được thêm vào dạng góc
    (angular margin m), giúp biên giới quyết định sắc hơn trong không gian hypersphere.

Cơ chế:
    1. Chuẩn hóa L2 embedding đầu vào và weight vector.
    2. Tính cosine similarity: cos(θ) = W^T · x
    3. Thêm margin: cos(θ + m)
    4. Scale bằng hệ số s rồi tính cross-entropy loss.

Args:
    in_features  (int)  : Chiều của embedding đầu vào (phải khớp với PalmNet, mặc định 128).
    num_classes  (int)  : Số lượng class (số người trong tập train).
    scale        (float): Hệ số nhân scale s. Thường dùng 32–64. Mặc định: 64.0
    margin       (float): Angular margin m (radian). Mặc định: 0.5 (~28.6°)

Input:
    embeddings : Tensor (B, in_features) — đầu ra của backbone (đã normalize L2)
    labels     : Tensor (B,)             — class index tương ứng

Output:
    Scalar loss (cross-entropy sau khi thêm angular margin)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        # Ma trận weight cho mỗi class — mỗi hàng là 1 prototype embedding
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Tính trước các hằng số để tránh recompute mỗi bước
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # Ngưỡng: khi cos(θ) < cos(π - m) thì θ + m > π,
        # lúc đó dùng fallback tuyến tính thay vì cos(θ + m) để giữ monotonicity
        self.threshold = math.cos(math.pi - margin)
        self.safe_margin = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings : (B, in_features) — embedding từ backbone
            labels     : (B,)             — label dạng long integer

        Returns:
            loss (scalar)
        """
        # Bước 1: Normalize embedding và weight để làm việc trên hypersphere
        emb_norm    = F.normalize(embeddings, p=2, dim=1)   # (B, D)
        weight_norm = F.normalize(self.weight, p=2, dim=1)  # (C, D)

        # Bước 2: cos(θ_j) cho mọi class j
        cosine = F.linear(emb_norm, weight_norm)  # (B, C)

        # Bước 3: sin(θ) từ pythagorean identity
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-9))  # (B, C)

        # Bước 4: cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m  # (B, C)

        # Bước 5: Fallback nếu θ + m > π (đảm bảo monotonicity)
        phi = torch.where(cosine > self.threshold, phi, cosine - self.safe_margin)

        # Bước 6: Chỉ thêm margin vào class ground-truth
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # (B, C)

        # Bước 7: Scale và cross-entropy
        logits = logits * self.scale
        loss = F.cross_entropy(logits, labels.long())

        return loss
