import os
import random
from itertools import combinations

from PIL import Image

import torch
from torch.utils.data import Dataset


class ThresholdDataset(Dataset):
    def __init__(self, root_dir, num_negative_per_class=5, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # LOAD ALL IMAGES
        self.class_to_images = {}

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            images = []

            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(class_dir, file_name))

            if len(images) >= 2:
                self.class_to_images[class_name] = images

        self.class_names = list(self.class_to_images.keys())

        # BUILD PAIRS
        self.pairs = []

        # POSITIVE PAIRS
        for class_name, images in self.class_to_images.items():
            positive_pairs = list(combinations(images, 2))

            for img1, img2 in positive_pairs:
                self.pairs.append((img1, img2, 1))

        # NEGATIVE PAIRS
        for class_name, images in self.class_to_images.items():

            other_classes = [
                c for c in self.class_names
                if c != class_name
            ]

            for img1 in images:
                for _ in range(num_negative_per_class):
                    neg_class = random.choice(other_classes)
                    img2 = random.choice(self.class_to_images[neg_class])
                    self.pairs.append((img1, img2, 0))

        random.shuffle(self.pairs)

        print(f"\nTotal pairs : {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2, torch.tensor(label, dtype=torch.long))
