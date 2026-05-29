"""
Biến đổi tăng cường tương phản cục bộ bằng CLAHE (OpenCV).

Chuyển ảnh về grayscale (nếu cần) và áp dụng CLAHE để cải thiện độ tương phản, giúp làm rõ các chi tiết như texture lòng bàn tay.
Nhận và trả về ảnh dạng numpy.
"""

import cv2
import numpy as np
from PIL import Image


class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(5,5)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)

        if img_np.ndim == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

        img_np = clahe.apply(img_np)

        return Image.fromarray(img_np)