import os
import random
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

class TripletDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.root_1 = os.path.join(root, "session1")
        self.root_2 = os.path.join(root, "session2")

        files_1 = sorted(os.listdir(self.root_1))
        files_2 = sorted(os.listdir(self.root_2))

        self.images_1 = []
        self.images_2 = []
        self.labels = []

        self.label_to_indices = defaultdict(list)

        num_per_person = 10

        for idx in range(len(files_1)):

            img1 = os.path.join(self.root_1, files_1[idx])
            img2 = os.path.join(self.root_2, files_2[idx])

            label = idx // num_per_person

            self.images_1.append(img1)
            self.images_2.append(img2)
            self.labels.append(label)
            self.label_to_indices[label].append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        anchor = Image.open(self.images_1[idx])
        anchor_label = self.labels[idx]

        pos_candidates = self.label_to_indices[anchor_label]
        pos_idx = random.choice([i for i in pos_candidates if i != idx])
        positive = Image.open(self.images_2[pos_idx])

        negative_label = random.choice([l for l in self.unique_labels if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative = Image.open(self.images_1[negative_idx])

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative