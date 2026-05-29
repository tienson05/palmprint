import os
import json
import argparse
from datetime import datetime

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, roc_auc_score

from src.datasets.threshold_dataset import ThresholdDataset
from src.model.palm_net import PalmNet
from src.transforms.transform_pipeline import eval_transform


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./palmnet_arcface_best.pth")
    parser.add_argument("--data_dir", type=str, default="../palm-dataset/val/val")

    parser.add_argument("--batch_size", "-b", type=int,   default=64)
    parser.add_argument("--num_workers","-n",type=int,   default=4)

    # If training on Kaggle or Colab
    # args, unknown = parser.parse_known_args()
    # # Bỏ qua các argument lạ (như -f kernel.json)
    # if unknown:
    #     print(f"Ignored unknown arguments: {unknown}")
    #
    # return args

    return parser.parse_args()

# LOAD MODEL
def load_model(model_path, device):
    model = PalmNet()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"\nLoaded model : {model_path}")

    return model

@torch.no_grad()
def extract_scores(model, loader, device):

    all_scores = []
    all_labels = []

    for img1, img2, labels in loader:

        img1 = img1.to(device)
        img2 = img2.to(device)

        emb1 = model(img1)
        emb2 = model(img2)

        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)

        scores = F.cosine_similarity(emb1, emb2)

        all_scores.extend(scores.cpu().numpy())
        all_labels.extend(labels.numpy())

    return (
        np.array(all_scores),
        np.array(all_labels)
    )

def compute_metrics(scores, labels, threshold):

    preds = (scores >= threshold).astype(int)

    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    far = fp / (fp + tn + 1e-12)
    frr = fn / (fn + tp + 1e-12)

    acc = (tp + tn) / len(labels)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "far": float(far),
        "frr": float(frr),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

def find_eer(scores, labels):

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))

    eer = fpr[idx]
    threshold = thresholds[idx]

    return eer, threshold

def sweep_thresholds(scores, labels, start=0.0, end=1.0, step=0.01):
    results = []

    thresholds = np.arange(start, end + step, step)

    for th in thresholds:
        metrics = compute_metrics(scores, labels, th)
        results.append(metrics)

    return results

def find_threshold_by_far(results, target_far=0.001):
    candidates = [
        r for r in results
        if r["far"] <= target_far
    ]

    if len(candidates) == 0:
        return None

    best = min(
        candidates,
        key=lambda x: x["frr"]
    )

    return best


def main():
    args = get_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(args.model_path, device)

    # DATASET
    dataset = ThresholdDataset(root_dir=args.data_dir, transform=eval_transform)
    print(f"\nDataset size : {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )


    scores, labels = extract_scores(model, loader, device)
    print(f"\nTotal pairs : {len(scores)}")

    os.makedirs("/kaggle/working/eval_results", exist_ok=True)

    np.save("/kaggle/working/eval_results/scores.npy", scores)

    np.save("/kaggle/working/eval_results/labels.npy", labels)

    roc_auc = roc_auc_score(labels, scores)
    print(f"\nROC-AUC : {roc_auc:.6f}")

    eer, eer_threshold = find_eer(scores, labels)
    print(f"\nEER           : {eer:.6f}")
    print(f"EER Threshold : {eer_threshold:.6f}")

    results = sweep_thresholds(scores, labels, start=0.0, end=1.0, step=0.01)

    far_targets = [0.01, 0.001, 0.0001]

    summary = {}

    for target_far in far_targets:
        best = find_threshold_by_far(results, target_far)

        if best is None:
            continue

        print("\n====================================")
        print(f"TARGET FAR <= {target_far}")
        print("====================================")

        print(json.dumps(best, indent=4))

        summary[f"far_{target_far}"] = best

    # SAVE JSON
    save_dict = {
        "saved_at": str(datetime.now()),
        "model_path": args.model_path,
        "roc_auc": float(roc_auc),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "summary": summary,
        "all_results": results,
    }

    save_path = os.path.join(
        "/kaggle/working/eval_results",
        "threshold_eval.json"
    )

    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=4)

    print(f"\nSaved : {save_path}")

if __name__ == "__main__":
    main()
