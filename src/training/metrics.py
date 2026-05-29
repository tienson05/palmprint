import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import auc

# COMPUTE EMBEDDINGS
def compute_embeddings(model, dataloader, device):
    embeddings = []
    labels = []
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            emb = model(img)
            embeddings.append(emb.cpu())
            labels.extend(label.numpy())

    embeddings = torch.cat(embeddings)
    return embeddings, np.array(labels)

# BUILD GALLERY - Tính mean các embedding cùng label để tạo 1 vector đại diện
def build_gallery(embeddings, labels):
    gallery = {}
    label_dict = defaultdict(list)

    for emb, label in zip(embeddings, labels):
        label_dict[label].append(emb)

    for label in label_dict:
        gallery[label] = torch.stack(label_dict[label]).mean(0)

    return gallery

# CREATE PAIRS - Với mỗi ảnh probe ta so sánh với all class trong gallery
def create_pairs(probe_embs, probe_labels, gallery):
    # stack gallery
    gallery_labels = list(gallery.keys())
    gallery_mat = torch.stack([gallery[k] for k in gallery_labels])  # [C, D]

    similarities = []
    labels = []

    for emb, label in zip(probe_embs, probe_labels):
        # cosine similarity vectorized
        sim_vec = torch.mv(gallery_mat, emb)  # [C]

        similarities.append(sim_vec)
        labels.extend([1 if label == g else 0 for g in gallery_labels])

    return np.concatenate(similarities), np.array(labels)

# FAR - False Accept Rate: nhận nhầm người lạ là đúng
# FRR - False Reject Rate: từ chối đúng người
def compute_far_frr(similarities, labels, threshold):
    preds = similarities > threshold

    TP = np.sum((preds & (labels == 1)))
    TN = np.sum((~preds & (labels == 0)))
    FP = np.sum((preds & (labels == 0)))
    FN = np.sum((~preds & (labels == 1)))

    FAR = FP / (FP + TN + 1e-8)
    FRR = FN / (FN + TP + 1e-8)

    return FAR, FRR, TP, TN, FP, FN

# EER - Equal Error Rate: điểm mà FAR = FRR tức là mức cân bằng giữa 2 lỗi FAR và FRR
def compute_eer(similarities, labels):
    similarities = np.array(similarities)
    labels = np.array(labels)

    # sort by score
    idx = np.argsort(similarities)
    sims = similarities[idx]
    labs = labels[idx]

    # cumulative counts
    P = np.sum(labs == 1)
    N = np.sum(labs == 0)

    TP = P
    FP = N

    best_diff = 1e9
    eer = 0
    threshold = 0

    i = 0

    for t in sims:
        while i < len(sims) and sims[i] <= t:
            if labs[i] == 1:
                TP -= 1
            else:
                FP -= 1
            i += 1

        FN = P - TP
        TN = N - FP

        FAR = FP / (FP + TN + 1e-8)
        FRR = FN / (FN + TP + 1e-8)

        diff = abs(FAR - FRR)

        if diff < best_diff:
            best_diff = diff
            eer = (FAR + FRR) / 2
            threshold = t

    return eer, threshold

# ACCURACY
def compute_accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN + 1e-8)

# TPR @ FAR: tìm TPR cao nhất nhưng vẫn đảm bảo FAR ≤ target
# FAR - False Accept Rate: nhận nhầm người lạ là đúng
def compute_tpr_at_far(similarities, labels, target_far=1e-3):
    thresholds = np.linspace(similarities.min(), similarities.max(), 2000)

    best_tpr = 0

    for t in thresholds:
        FAR, FRR, TP, TN, FP, FN = compute_far_frr(similarities, labels, t)

        if FAR <= target_far:
            TPR = TP / (TP + FN + 1e-8)
            best_tpr = max(best_tpr, TPR)

    return best_tpr

# ROC curve
# FAR - False Accept Rate: nhận nhầm người lạ là đúng
# TPR - True Positive Rate: nhận đúng người thật
def compute_roc(similarities, labels):
    similarities = np.array(similarities)
    labels = np.array(labels)

    # sort descending
    idx = np.argsort(-similarities)
    sims = similarities[idx]
    labs = labels[idx]

    P = np.sum(labs == 1)
    N = np.sum(labs == 0)

    TP = 0
    FP = 0

    tprs = []
    fars = []

    prev_score = None

    for i in range(len(sims)):
        if labs[i] == 1:
            TP += 1
        else:
            FP += 1

        TPR = TP / (P + 1e-8)
        FAR = FP / (N + 1e-8)

        if sims[i] != prev_score:
            tprs.append(TPR)
            fars.append(FAR)
            prev_score = sims[i]

    return np.array(fars), np.array(tprs), sims

def eval_pipeline(model, gallery_loader, probe_loader, device, target_far=1e-3):
    model.eval()

    # gallery embeddings
    gallery_embs, gallery_labels = compute_embeddings(model, gallery_loader, device)

    # probe embeddings
    probe_embs, probe_labels = compute_embeddings(model, probe_loader, device)

    # build gallery
    gallery = build_gallery(gallery_embs, gallery_labels)

    # create pairs
    similarities, labels = create_pairs(probe_embs, probe_labels, gallery)

    # EER
    eer, threshold = compute_eer(similarities, labels)

    # FAR FRR
    FAR, FRR, TP, TN, FP, FN = compute_far_frr(similarities, labels, threshold)

    # Accuracy
    accuracy = compute_accuracy(TP, TN, FP, FN)

    # ROC
    fars, tprs, _ = compute_roc(similarities, labels)
    roc_auc = auc(fars, tprs)

    # TPR @ FAR
    tpr_at_far = compute_tpr_at_far(similarities, labels, target_far=target_far)

    # mean positive negative similarity
    pos_sims = similarities[labels == 1]
    neg_sims = similarities[labels == 0]
    mean_pos = pos_sims.mean()
    mean_neg = neg_sims.mean()
    gap = mean_pos - mean_neg

    return {
        "accuracy": accuracy, "eer": eer, "threshold": threshold,
        "far": FAR, "frr": FRR, "roc_auc": roc_auc,
        "tpr_at_far": tpr_at_far,
        "mean_pos": mean_pos, "mean_neg": mean_neg, "gap": gap,
        "tp": TP, "tn": TN, "fp": FP, "fn": FN
    }