import os

from PIL import Image
from torch.utils.data import Dataset

class ArcFaceDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.num_per_person = 10
        self.samples = []   # list of (image_path, label)

        # Phát hiện loại cấu trúc thư mục
        session_dirs = []
        for session in ["session1", "session2"]:
            sd = os.path.join(root, session)
            if os.path.isdir(sd):
                session_dirs.append(sd)

        if session_dirs:
            for session_dir in session_dirs:
                files = sorted(os.listdir(session_dir))
                for idx, fname in enumerate(files):
                    path = os.path.join(session_dir, fname)
                    try:
                        img_id = int(fname.split('.')[0])
                        label = (img_id - 1) // self.num_per_person
                    except:
                        label = 0
                    self.samples.append((path, label))
        else:
            persons = sorted([
                p for p in os.listdir(root)
                if os.path.isdir(os.path.join(root, p))
            ])
            for label, person in enumerate(persons):
                person_dir = os.path.join(root, person)
                for fname in sorted(os.listdir(person_dir)):
                    path = os.path.join(person_dir, fname)
                    self.samples.append((path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"Không tìm thấy ảnh nào trong: {root}")

        self.num_classes = len(set(label for _, label in self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
