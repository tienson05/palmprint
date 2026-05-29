import os
import time
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


# Hàm lấy ROI lòng bàn tay
def crop_palm_roi(image, landmarks, roi_size=224):
    h, w, _ = image.shape

    def to_xy(lm):
        return np.array([lm.x * w, lm.y * h])

    L0 = to_xy(landmarks[0])
    L5 = to_xy(landmarks[5])
    L9 = to_xy(landmarks[9])
    L17 = to_xy(landmarks[17])

    # STEP 1: vector L5L17
    v = L17 - L5
    s = np.linalg.norm(v)

    # STEP 2: normals
    n1 = np.array([-v[1], v[0]])
    n2 = np.array([v[1], -v[0]])

    n1 = n1 / np.linalg.norm(n1) # triệt tiêu độ lớn, chỉ còn hướng
    n2 = n2 / np.linalg.norm(n2)

    # STEP 3: reference point
    Lref = (L0 + L9) / 2

    # STEP 4: midpoint
    M = (L5 + L17) / 2

    P1 = M + s * n1
    P2 = M + s * n2

    if np.linalg.norm(P1 - Lref) < np.linalg.norm(P2 - Lref):
        n = n1
    else:
        n = n2

    # STEP 5: ROI corners
    Qsrc = np.array([
        L5,
        L17,
        L17 + s * n,
        L5 + s * n
    ], dtype=np.float32)

    # STEP 6: destination square
    S = roi_size
    Qdst = np.array([
        [0, 0],
        [S-1, 0],
        [S-1, S-1],
        [0, S-1]
    ], dtype=np.float32)

    # STEP 7: perspective transform
    M = cv2.getPerspectiveTransform(Qsrc, Qdst)

    # STEP 8: warp ROI
    roi = cv2.warpPerspective(
        image,
        M,
        (S, S),
        flags=cv2.INTER_CUBIC
    )
    return roi

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)
# Paths
path = "../data/raw/IITD/Right Hand"  # or session2
out = "../data/processed/IITD/Right_Hand" # or session2
imgs = os.listdir(path)
pg_bar = tqdm(imgs)

count = 0
time_sum = 0
not_detected = []
not_readed =  []

for img_name in pg_bar:
    name = os.path.splitext(img_name)[0]
    start = time.time()

    img_path = os.path.join(path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        not_readed.append(img_name)
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    elapsed = time.time() - start

    time_sum += elapsed

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            roi = crop_palm_roi(image=img_rgb, landmarks=hand_landmarks.landmark, roi_size=224)
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out, name + ".png"), roi_bgr)
        count += 1
    else:
        not_detected.append(img_name)

print(f"Detected {count} images")
print(f"Total time: {time_sum:.4f} seconds")
print(f"Images aren't detected {not_detected}")
print(f"Images aren't read {not_readed}")

hands.close()