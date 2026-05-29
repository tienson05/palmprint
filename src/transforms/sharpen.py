"""
Biến đổi làm sắc nét ảnh đơn giản bằng OpenCV.

Áp dụng bộ lọc sharpen với xác suất `p` để làm nổi bật chi tiết nhỏ (ở đây là: texture lòng bàn tay).
Nhận và trả về ảnh dạng numpy.
"""

import random
import cv2
import numpy as np
from PIL import Image


class SharpenTransform:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img_np = np.array(img)
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            img_np = cv2.filter2D(img_np, -1, kernel)
            return Image.fromarray(img_np)
        return img