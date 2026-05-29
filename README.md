# 🖐️ Palmprint Recognition

Hệ thống nhận dạng lòng bàn tay (palmprint) sử dụng deep metric learning. Mô hình được xây dựng trên nền tảng ResNet tích hợp Squeeze-and-Excitation (SE) Block, huấn luyện bằng **ArcFace Loss** hoặc **Triplet Loss**, và đánh giá bằng các chỉ số sinh trắc học chuẩn (EER, ROC-AUC, FAR/FRR).

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt môi trường](#-cài-đặt-môi-trường)
- [Chuẩn bị dữ liệu](#-chuẩn-bị-dữ-liệu)
- [Huấn luyện](#-huấn-luyện)
- [Đánh giá mô hình](#-đánh-giá-mô-hình)
- [Kết quả](#-kết-quả)
- [Chi tiết kỹ thuật](#-chi-tiết-kỹ-thuật)

---

## 🔍 Tổng quan

Dự án giải quyết bài toán **xác minh danh tính (identity verification)** dựa trên ảnh lòng bàn tay, gồm các bước:

1. **Trích xuất ROI** — Dùng MediaPipe để phát hiện bàn tay và cắt vùng lòng bàn tay bằng phép biến đổi phối cảnh (perspective transform).
2. **Huấn luyện embedding** — Học vector đặc trưng 128 chiều từ ảnh lòng bàn tay.
3. **Xác minh** — So sánh cosine similarity giữa hai embedding để quyết định cùng hay khác người.

**Hai chế độ huấn luyện:**

| Loss Function | Phương pháp | Đặc điểm |
|---|---|---|
| `arcface` | Classification-based | Thêm angular margin vào không gian hypersphere, cho biên giới quyết định sắc hơn |
| `triplet` | Metric learning | Học trực tiếp khoảng cách: anchor gần positive, xa negative |

---

## 🧠 Kiến trúc mô hình

### PalmNet

Mạng CNN được tùy chỉnh từ ResNet, kết hợp SE Block để tăng cường chú ý theo kênh:

```
Input (B, 1, 224, 224)
    │
    ▼
Stem: Conv7×7 → BN → ReLU → MaxPool     → (B, 64, 56, 56)
    │
    ▼
Layer1: ResBlock(64→64) × 2              → (B, 64, 56, 56)
    │
    ▼
Layer2: ResBlock(64→128, stride=2) × 2   → (B, 128, 28, 28)
    │
    ▼
Layer3: ResBlock(128→256, stride=2) × 2  → (B, 256, 14, 14)
    │
    ▼
Layer4: ResBlock(256→512, stride=2) × 2  → (B, 512, 7, 7)
    │
    ▼
Head:  GAP → Flatten → Dropout(0.2) → FC(512→128)
    │
    ▼
L2 Normalize
    │
    ▼
Output (B, 128)  ← unit-norm embedding
```

### ResBlock + SE Block

Mỗi `ResBlock` tích hợp **Squeeze-and-Excitation** để học trọng số chú ý theo kênh:

```
Input
  ├─ Conv3×3 → BN → ReLU → Conv3×3 → BN → SE Block ─┐
  │                                                    + → ReLU → Output
  └─ Shortcut (Identity hoặc Conv1×1 projection) ────┘
```

**SE Block:**
- **Squeeze**: Global Average Pooling → vector `(B, C)`
- **Excitation**: FC → ReLU → FC → Sigmoid → scale `(B, C, 1, 1)`
- Nhân channel-wise với feature map gốc để tái hiệu chỉnh

### ArcFace Loss

Thêm angular margin `m` vào góc giữa embedding và class prototype:

```
cos(θ + m) = cos(θ)·cos(m) − sin(θ)·sin(m)
logits = scale × [margin applied on GT class]
loss = CrossEntropy(logits, labels)
```

Mặc định: `scale=64.0`, `margin=0.5` (~28.6°)

---

## 📁 Cấu trúc dự án

```
palmprint/
├── data/
│   ├── raw/              # Ảnh gốc (IITD và Tongji)
│   ├── processed/        # Ảnh đã trích xuất ROI
│   └── splits/           # Train / Val / Test splits
│       ├── train/
│       │   ├── session1/
│       │   └── session2/
│       ├── valid/
│       └── test/
│
├── docs/
│   └── images/           # Ảnh kết quả (loss, metrics, confusion, distance)
│
├── models/               # File .pth (checkpoint tốt nhất)
├── results/              # Metrics JSON, score arrays
├── runs/                 # TensorBoard logs
│
├── scripts/
│   ├── extract_roi.py    # Trích xuất ROI lòng bàn tay bằng MediaPipe
│   └── splits.py         # Chia dataset thành train/val/test
│
├── src/
│   ├── datasets/
│   │   ├── arcface4train_dataset.py   # Dataset cho ArcFace training
│   │   ├── triplet4train_dataset.py   # Dataset cho Triplet training
│   │   ├── eval_dataset.py            # Dataset đánh giá (query/gallery)
│   │   └── threshold_dataset.py       # Dataset tạo cặp ảnh để sweep threshold
│   │
│   ├── model/
│   │   ├── palm_net.py       # Kiến trúc PalmNet chính
│   │   ├── res_block.py      # Residual Block tích hợp SE
│   │   ├── se_block.py       # Squeeze-and-Excitation Block
│   │   └── arcface_loss.py   # ArcFace Loss
│   │
│   ├── training/
│   │   ├── train.py           # Script huấn luyện chính
│   │   ├── metrics.py         # Tính EER, FAR, FRR, ROC-AUC, ...
│   │   ├── compare.py         # So sánh hai mô hình
│   │   ├── sweep_threshold.py # Tìm ngưỡng tối ưu trên tập val
│   │   └── utils.py           # Lưu metrics JSON
│   │
│   └── transforms/
│       ├── transform_pipeline.py  # Pipeline augmentation train/eval
│       ├── clahe.py               # CLAHE contrast enhancement
│       └── sharpen.py             # Sharpening transform
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt môi trường

**Yêu cầu:** Python 3.10, CUDA (khuyến nghị)

```bash
# 1. Tạo virtual environment
py -3.10 -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Cài đặt dependencies
pip install -r requirements.txt
```

**Các thư viện chính:**

| Thư viện | Mục đích |
|---|---|
| `torch`, `torchvision` | Deep learning framework |
| `mediapipe` | Phát hiện landmark bàn tay |
| `opencv-python` | Xử lý ảnh, CLAHE, perspective transform |
| `scikit-learn` | Tính ROC-AUC, metrics |
| `tensorboard` | Theo dõi quá trình huấn luyện |
| `tqdm` | Progress bar |
| `Pillow` | Đọc/ghi ảnh |
| `numpy` | Tính toán số học |

---

## 📦 Chuẩn bị dữ liệu

### Bước 1: Trích xuất ROI lòng bàn tay

Sửa đường dẫn trong `scripts/extract_roi.py` và chạy:

```bash
# Chỉnh sửa biến path và out trong file
python scripts/extract_roi.py
```

Script dùng **MediaPipe** để phát hiện 21 landmark bàn tay, sau đó thực hiện **perspective transform** dựa trên 4 điểm khung lòng bàn tay (L5, L17) để cắt vùng ROI 224×224 pixel, bất biến với góc quay và tỉ lệ.

### Bước 2: Chia tập dữ liệu

```bash
python scripts/splits.py
```

Tỉ lệ chia mặc định:
- **Train**: 100% của Tongji
- **Val**: 50% identities của IITD
- **Test**: 50% identities của IITD

Cấu trúc output:
```
data/splits/
├── train/
│   ├── session1/   ← ảnh chụp lần 1
│   └── session2/   ← ảnh chụp lần 2 (cùng người, khác phiên)
├── valid/
└── test/
```

---

## 🚀 Huấn luyện

Chạy từ thư mục gốc của dự án:

```bash
# Huấn luyện với ArcFace (khuyến nghị)
python -m src.training.train \
    --loss arcface \
    --train_path data/splits/train \
    --val_path data/splits/valid \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --scale 64.0 \
    --margin 0.5

# Huấn luyện với Triplet Loss
python -m src.training.train \
    --loss triplet \
    --train_path data/splits/train \
    --val_path data/splits/valid \
    --epochs 50 \
    --batch_size 64 \
    --margin 0.5
```

### Các tham số huấn luyện

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--loss`, `-l` | `arcface` | Hàm loss: `arcface` hoặc `triplet` |
| `--train_path` | `data/splits/train` | Thư mục dữ liệu train |
| `--val_path` | `data/splits/val` | Thư mục dữ liệu validation |
| `--save_dir`, `-s` | `models/` | Thư mục lưu checkpoint |
| `--runs_dir`, `-r` | `runs/` | Thư mục lưu TensorBoard log |
| `--model_name` | `palmnet_arcface` | Tên file checkpoint |
| `--lr` | `1e-4` | Learning rate |
| `--batch_size`, `-b` | `64` | Batch size |
| `--epochs`, `-e` | `50` | Số epoch |
| `--num_workers`, `-n` | `4` | Số worker cho DataLoader |
| `--scale` | `64.0` | ArcFace scale factor `s` |
| `--margin`, `-m` | `0.5` | Angular margin (ArcFace) hoặc distance margin (Triplet) |

### Theo dõi với TensorBoard

```bash
tensorboard --logdir runs/
```

Các metric được log:
- `Loss/train_step` — Loss mỗi batch
- `Loss/train` — Loss trung bình mỗi epoch
- `Metric/EER`, `Metric/ROC_AUC`, `Metric/Accuracy`
- `Metric/FAR`, `Metric/FRR`, `Metric/TPR_at_FAR`
- `Distance/Positive`, `Distance/Negative`, `Distance/Gap`
- `Confusion/TP`, `Confusion/TN`, `Confusion/FP`, `Confusion/FN`

### Checkpoint

Mô hình tốt nhất (EER thấp nhất) được lưu tự động tại `models/` với tên theo format:

```
palmnet_arcface_bs64_lr0.0001_m0.5_s64.0_epoch50_cosineLR.pth
```

Kèm file JSON chứa metrics tương ứng.

---

## 📊 Đánh giá mô hình

### So sánh hai mô hình

```bash
python -m src.training.compare \
    --model1 models/model_a.pth \
    --model2 models/model_b.pth \
    --val_path data/splits/valid
```

Output bảng so sánh các chỉ số: EER, ROC-AUC, Accuracy, FAR, FRR, TPR@FAR, embedding gap.

### Sweep Threshold

Tìm ngưỡng cosine similarity tối ưu trên tập validation:

```bash
python -m src.training.sweep_threshold \
    --model_path models/best_model.pth \
    --data_dir data/splits/valid
```

Kết quả bao gồm:
- EER và ngưỡng tương ứng
- Ngưỡng tối ưu ứng với `FAR ≤ 0.01`, `FAR ≤ 0.001`, `FAR ≤ 0.0001`
- ROC-AUC score

---

## 📈 Kết quả

Kết quả trực quan từ quá trình huấn luyện (xem thư mục `docs/images/`):

| Biểu đồ | Nội dung |
|---|---|
| `loss.png` | Train loss theo epoch |
| `metric.png` | EER, ROC-AUC, Accuracy, FAR, FRR, TPR@FAR |
| `distance.png` | Cosine similarity trung bình của cặp positive/negative và khoảng cách gap |
| `confusion.png` | TP, TN, FP, FN theo epoch |

---

## 🔧 Chi tiết kỹ thuật

### Pipeline tiền xử lý ảnh

**Train transform:**
```
Resize(224×224)
→ RandomAffine(degrees=8, translate=2%, scale=98–102%)
→ ColorJitter(brightness=0.15, contrast=0.15)
→ SharpenTransform(p=0.5)
→ Grayscale(1 channel)
→ CLAHE(clip_limit=2.0, tile=(5×5))
→ ToTensor()
→ Normalize(mean=0.5, std=0.5)
```

**Eval transform:**
```
Resize(224×224)
→ Grayscale(1 channel)
→ CLAHE(clip_limit=2.0, tile=(5×5))
→ ToTensor()
→ Normalize(mean=0.5, std=0.5)
```

> **CLAHE** (Contrast Limited Adaptive Histogram Equalization) giúp cải thiện độ tương phản cục bộ, làm nổi bật các chi tiết texture đặc trưng của lòng bàn tay.

### Chiến lược đánh giá (Gallery-Probe)

1. **Gallery**: Tính embedding trung bình (mean pooling) của tất cả ảnh cùng một người → 1 vector đại diện/người.
2. **Probe (Query)**: Mỗi ảnh query được so sánh cosine similarity với toàn bộ gallery.
3. **Tạo cặp**: Mỗi cặp (probe, gallery) được gán nhãn `1` (cùng người) hoặc `0` (khác người).
4. **Tính metrics** từ danh sách similarity scores và labels.

### Các chỉ số sinh trắc học

| Chỉ số | Ý nghĩa |
|---|---|
| **EER** (Equal Error Rate) | Điểm mà FAR = FRR — càng thấp càng tốt |
| **FAR** (False Accept Rate) | Tỉ lệ chấp nhận nhầm người lạ |
| **FRR** (False Reject Rate) | Tỉ lệ từ chối nhầm đúng người |
| **ROC-AUC** | Diện tích dưới đường ROC — càng cao càng tốt |
| **TPR@FAR=0.1%** | True Positive Rate khi FAR ≤ 0.001 |
| **Gap** | `mean_pos_sim − mean_neg_sim` — độ tách biệt embedding |

### Learning Rate Schedule

Sử dụng **Cosine Annealing LR**:
- `T_max = epochs`
- `eta_min = 1e-6`

Kết hợp với **Mixed Precision Training** (`torch.amp.autocast`) để tăng tốc và giảm bộ nhớ GPU.