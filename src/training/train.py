import argparse
import os
import sys
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets.arcface4train_dataset import ArcFaceDataset
from src.datasets.eval_dataset import EvalDataset
from src.datasets.triplet4train_dataset import TripletDataset
from src.model.arcface_loss import ArcFaceLoss
from src.model.palm_net import PalmNet
from src.training.metrics import eval_pipeline
from src.training.utils import save_best_metrics
from src.transforms.transform_pipeline import train_transform, eval_transform

def get_args():
    parser = argparse.ArgumentParser()

    # Loss function
    parser.add_argument("--loss", "-l",      type=str,   default="arcface")

    # Paths
    parser.add_argument("--train_path",      type=str,   default="../../data/splits/train")
    parser.add_argument("--val_path",        type=str,   default="../../data/splits/val")
    parser.add_argument("--save_dir",  "-s", type=str,   default="../../models/")
    parser.add_argument("--runs_dir",  "-r", type=str,   default="../../runs/")
    parser.add_argument("--model_name",      type=str,   default="palmnet_arcface")

    # Training hyper-params
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--batch_size", "-b",type=int,   default=64)
    parser.add_argument("--epochs",     "-e",type=int,   default=50)
    parser.add_argument("--num_workers","-n",type=int,   default=4)
    parser.add_argument("--colour",     "-c",type=str,   default="green")

    # ArcFace specific
    parser.add_argument("--scale",  type=float, default=64.0, help="ArcFace scale factor s")

    # Common
    parser.add_argument("--margin", "-m", type=float, default=0.5)

    # If training on Kaggle or Colab
    # args, unknown = parser.parse_known_args()
    # # Bỏ qua các argument lạ (như -f kernel.json)
    # if unknown:
    #     print(f"Ignored unknown arguments: {unknown}")
    #
    # return args

    return parser.parse_args()

def main():
    args = get_args()

    loss_function = args.loss.lower()
    if loss_function != 'arcface' and loss_function != 'triplet':
        print("Please enter loss function again!")
        return

    # Device
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"   GPU   : {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.runs_dir, f"arcface_{timestamp}")
    writer = SummaryWriter(log_dir)

    # Model
    model = PalmNet().to(device)

    # Datasets & Loss & Optimizer
    if loss_function == 'arcface':
        train_dataset = ArcFaceDataset(root=args.train_path, transform=train_transform)
        num_classes = train_dataset.num_classes
        print(f"Tập train : {len(train_dataset)} ảnh  |  {num_classes} classes\n")
        criterion = ArcFaceLoss(in_features=128, num_classes=num_classes, scale=args.scale, margin=args.margin).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(criterion.parameters()),
            lr=args.lr,
        )
    else:
        train_dataset = TripletDataset(root=args.train_path, transform=train_transform)
        criterion = nn.TripletMarginLoss(margin=args.margin)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    query_dataset   = EvalDataset(args.val_path, eval_transform, mode="query")
    gallery_dataset = EvalDataset(args.val_path, eval_transform, mode="gallery")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,   # tránh batch size 1 gây lỗi BatchNorm
    )

    query_loader = DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    gallery_loader = DataLoader(
        gallery_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    scaler = torch.amp.GradScaler('cuda')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Training Loop
    best_eer = float("inf")
    num_iter = len(train_loader)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        if loss_function == 'arcface':
            criterion.train()

        pbar = tqdm(train_loader, colour=args.colour, file=sys.stdout)

        if loss_function == 'arcface':
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    embeddings = model(images) # (B, 128), đã L2-normalize trong PalmNet
                    loss = criterion(embeddings, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                # Log per step
                writer.add_scalar("Loss/train_step", loss.item(), epoch * num_iter + batch_idx)
                pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} | Loss: {loss.item():.4f}")
        else:
            for batch_idx, (anchor, positive, negative) in enumerate(pbar):
                anchor = anchor.to(device, non_blocking=True)
                positive = positive.to(device, non_blocking=True)
                negative = negative.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    emb_a = model(anchor)
                    emb_p = model(positive)
                    emb_n = model(negative)

                    loss = criterion(emb_a, emb_p, emb_n)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                # Log per step
                writer.add_scalar("Loss/train_step", loss.item(), epoch * num_iter + batch_idx)
                pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} | Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)

        # Evaluation
        model.eval()
        metrics = eval_pipeline(model, gallery_loader, query_loader, device)

        eer         = metrics["eer"]
        threshold   = metrics["threshold"]
        roc_auc     = metrics["roc_auc"]
        accuracy    = metrics["accuracy"]
        far         = metrics["far"]
        frr         = metrics["frr"]
        tpr_at_far  = metrics["tpr_at_far"]
        mean_pos    = metrics["mean_pos"]
        mean_neg    = metrics["mean_neg"]
        gap         = metrics["gap"]
        tp          = metrics["tp"]
        tn          = metrics["tn"]
        fp          = metrics["fp"]
        fn          = metrics["fn"]

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Biometric metrics
        writer.add_scalar("Metric/EER",       eer,        epoch)
        writer.add_scalar("Metric/Threshold", threshold,  epoch)
        writer.add_scalar("Metric/ROC_AUC",   roc_auc,   epoch)
        writer.add_scalar("Metric/Accuracy",  accuracy,  epoch)
        writer.add_scalar("Metric/FAR",       far,       epoch)
        writer.add_scalar("Metric/FRR",       frr,       epoch)
        writer.add_scalar("Metric/TPR_at_FAR",tpr_at_far,epoch)

        # Embedding distance
        writer.add_scalar("Distance/Positive", mean_pos, epoch)
        writer.add_scalar("Distance/Negative", mean_neg, epoch)
        writer.add_scalar("Distance/Gap",      gap,      epoch)

        # Confusion matrix counts
        writer.add_scalar("Confusion/TP", tp, epoch)
        writer.add_scalar("Confusion/TN", tn, epoch)
        writer.add_scalar("Confusion/FP", fp, epoch)
        writer.add_scalar("Confusion/FN", fn, epoch)

        # Console
        print(f"\n{'=' * 80}")
        print(f"  Epoch [{epoch + 1}/{args.epochs}]")
        print(f"{'─' * 80}")
        print(f"  Train Loss  : {train_loss:.6f}")
        print(f"{'─' * 80}")
        print(f"  EER         : {eer:.6f}    Threshold  : {threshold:.6f}")
        print(f"  ROC-AUC     : {roc_auc:.6f}    Accuracy   : {accuracy:.6f}")
        print(f"  FAR         : {far:.6f}    FRR        : {frr:.6f}")
        print(f"  TPR@FAR=1e-3: {tpr_at_far:.6f}")
        print(f"{'─' * 80}")
        print(f"  Sim Pos (↑) : {mean_pos:.6f}    Sim Neg (↓): {mean_neg:.6f}    Gap: {gap:.6f}")
        print(f"{'─' * 80}")
        print(f"  TP={tp:<6}  TN={tn:<6}  FP={fp:<6}  FN={fn:<6}")
        print(f"{'=' * 80}\n")

        scheduler.step()

        # Checkpoint
        if eer < best_eer:
            best_eer = eer
            if loss_function == 'arcface':
                base_filename = (
                    f"{args.model_name}"
                    f"_bs{args.batch_size}"
                    f"_lr{args.lr}"
                    f"_m{args.margin}"
                    f"_s{args.scale}"
                    f"_epoch{args.epochs}"
                )
            else:
                base_filename = (
                    f"{args.model_name}"
                    f"_bs{args.batch_size}"
                    f"_lr{args.lr}"
                    f"_m{args.margin}"
                    f"_epoch{args.epochs}"
                )
            save_path = os.path.join(args.save_dir, base_filename + "_cosineLR.pth")
            torch.save(model.state_dict(), save_path)

            # Save json metrics
            save_best_metrics(save_dir=args.save_dir, filename=base_filename + ".json",
                metrics=metrics, args=args, epoch=epoch
            )

            print(f"Best model saved! EER = {eer:.4f}\n")

    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()