import os
import random
import shutil


def split_dataset(session1_root, session2_root, output_root):

    files1 = sorted(os.listdir(session1_root))
    files2 = sorted(os.listdir(session2_root))

    num_persons = len(files1) // 10
    print(f"Number of persons: {num_persons}")

    splits = ["train", "valid", "test"]
    sessions = ["session1", "session2"]

    # tạo thư mục
    for split in splits:
        for session in sessions:
            os.makedirs(os.path.join(output_root, split, session), exist_ok=True)

    for pid in range(num_persons):

        imgs1 = files1[pid*10:(pid+1)*10]
        imgs2 = files2[pid*10:(pid+1)*10]

        idxs = list(range(10))
        random.shuffle(idxs)

        val_idx = idxs[:2]
        test_idx = idxs[2:4]
        train_idx = idxs[4:]

        for i in train_idx:
            shutil.copy(
                os.path.join(session1_root, imgs1[i]),
                os.path.join(output_root, "train", "session1", imgs1[i])
            )
            shutil.copy(
                os.path.join(session2_root, imgs2[i]),
                os.path.join(output_root, "train", "session2", imgs2[i])
            )

        for i in val_idx:
            shutil.copy(
                os.path.join(session1_root, imgs1[i]),
                os.path.join(output_root, "valid", "session1", imgs1[i])
            )
            shutil.copy(
                os.path.join(session2_root, imgs2[i]),
                os.path.join(output_root, "valid", "session2", imgs2[i])
            )

        for i in test_idx:
            shutil.copy(
                os.path.join(session1_root, imgs1[i]),
                os.path.join(output_root, "test", "session1", imgs1[i])
            )
            shutil.copy(
                os.path.join(session2_root, imgs2[i]),
                os.path.join(output_root, "test", "session2", imgs2[i])
            )

    print("Dataset split completed!")

if __name__ == "__main__":
    session1 = "../../data/processed/session1"
    session2 = "../../data/processed/session2"

    output = "../../data/splits"

    split_dataset(session1, session2, output)