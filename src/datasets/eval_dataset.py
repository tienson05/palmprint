import os
import random

import cv2
from torch.utils.data import Dataset

class EvalDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        mode="query",   # query hoặc gallery
        query_ratio=0.25,
        seed=42
    ):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        rng = random.Random(seed)

        persons = sorted(os.listdir(root_dir))

        for label, person in enumerate(persons):

            person_dir = os.path.join(root_dir, person)

            if not os.path.isdir(person_dir):
                continue

            images = sorted(os.listdir(person_dir))

            image_paths = [
                os.path.join(person_dir, img)
                for img in images
            ]

            # shuffle cố định bằng seed
            rng.shuffle(image_paths)

            num_query = max(1, int(len(image_paths) * query_ratio))

            query_paths = image_paths[:num_query]
            gallery_paths = image_paths[num_query:]

            if mode == "query":
                selected_paths = query_paths

            elif mode == "gallery":
                selected_paths = gallery_paths

            else:
                raise ValueError("mode must be query or gallery")

            for path in selected_paths:
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label