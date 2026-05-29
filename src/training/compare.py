import os
import json
import argparse
from prettytable import PrettyTable

import torch
from torch.utils.data import DataLoader

from src.model.palm_net import PalmNet
from src.datasets.eval_dataset import EvalDataset
from metrics import eval_pipeline
from src.transforms.transform_pipeline import eval_transform

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model1", type=str, required=True, help="Path to first model (.pth)")
    parser.add_argument("--model2", type=str, required=True, help="Path to second model (.pth)")
    parser.add_argument("--val_path", type=str, required=True, help="Validation dataset path")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="../../results")

    # If training on Kaggle or Colab
    # args, unknown = parser.parse_known_args()
    # # Bỏ qua các argument lạ (như -f kernel.json)
    # if unknown:
    #     print(f"Ignored unknown arguments: {unknown}")
    #
    # return args

    return parser.parse_args()


def evaluate_model(model_path, query_loader, gallery_loader, device):
    model = PalmNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    metrics = eval_pipeline(
        model=model,
        gallery_loader=gallery_loader,
        probe_loader=query_loader,
        device=device
    )

    return metrics


def save_metrics(metrics, save_path):
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4, default=float)

    print(f"Saved metrics: {save_path}")


def compare_metrics(metrics1, metrics2, name1, name2):
    table = PrettyTable()
    table.field_names = ["Metric", name1, name2, "Better"]

    compare_rules = {
        "eer": "min",
        "roc_auc": "max",
        "accuracy": "max",
        "far": "min",
        "frr": "min",
        "tpr_at_far": "max",
        "mean_pos": "min",
        "mean_neg": "max",
        "gap": "max",
    }

    for metric, rule in compare_rules.items():
        v1 = metrics1[metric]
        v2 = metrics2[metric]

        if rule == "min":
            better = name1 if v1 < v2 else name2
        else:
            better = name1 if v1 > v2 else name2

        table.add_row([metric, f"{v1:.6f}", f"{v2:.6f}", better])

    return table


def main():
    args = get_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing device: {device}\n")

    query_dataset = EvalDataset(args.val_path, eval_transform, mode="query")

    gallery_dataset = EvalDataset(args.val_path, eval_transform, mode="gallery")

    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print("=" * 80)
    print("Evaluating Model 1")

    metrics1 = evaluate_model(args.model1, query_loader, gallery_loader, device)

    print("\n" + "=" * 80)
    print("Evaluating Model 2")

    metrics2 = evaluate_model(args.model2, query_loader, gallery_loader, device)

    name1 = os.path.basename(args.model1).replace(".pth", "")
    name2 = os.path.basename(args.model2).replace(".pth", "")

    save_metrics(metrics1, os.path.join(args.save_dir, f"{name1}.json"))
    save_metrics(metrics2, os.path.join(args.save_dir, f"{name2}.json"))

    table = compare_metrics(metrics1, metrics2, name1, name2)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(table)

if __name__ == "__main__":
    main()