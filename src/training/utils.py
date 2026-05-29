import json
import os
from datetime import datetime

def save_best_metrics(save_dir, filename, metrics, args, epoch):
    os.makedirs(save_dir, exist_ok=True)

    save_data = {
        "epoch": epoch + 1,

        # Model config
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "margin": args.margin,

        # Best metrics
        "eer": float(metrics["eer"]),
        "threshold": float(metrics["threshold"]),
        "roc_auc": float(metrics["roc_auc"]),
        "accuracy": float(metrics["accuracy"]),
        "far": float(metrics["far"]),
        "frr": float(metrics["frr"]),
        "tpr_at_far": float(metrics["tpr_at_far"]),

        # Distance metrics
        "mean_pos": float(metrics["mean_pos"]),
        "mean_neg": float(metrics["mean_neg"]),
        "gap": float(metrics["gap"]),

        # Confusion matrix
        "tp": int(metrics["tp"]),
        "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),

        # Time
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    json_path = os.path.join(save_dir, filename)

    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=4)

    print(f"Best metrics saved to: {json_path}")