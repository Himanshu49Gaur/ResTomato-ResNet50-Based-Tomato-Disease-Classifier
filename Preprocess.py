import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# === Settings ===
train_path = "path to train data"
valid_path = "path to validation data"
preprocessed_dir = "path to save preprocessed data"
target_size = (224, 224)

# === Helper to Load Images and Labels ===
def load_images_from_folder(folder):
    images, labels = [], []
    print(f"\n🔍 Loading from: {folder}")
    for class_name in sorted(os.listdir(folder)):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue
        img_files = glob(os.path.join(class_path, '*.*'))
        for img_path in img_files:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                labels.append(class_name)
        print(f"✅ Loaded {len(img_files)} images for class '{class_name}'")
    return images, labels

# === Load All Images from Train and Valid ===
all_images, all_labels = [], []
for subdir in [train_dir, valid_dir]:
    imgs, lbls = load_images_from_folder(subdir)
    all_images.extend(imgs)
    all_labels.extend(lbls)

# === Preprocessing and Saving ===
os.makedirs(preprocessed_dir, exist_ok=True)
processed_count = 0
print("\n⚙️ Starting preprocessing and saving...\n")

for i, (img, label) in enumerate(tqdm(zip(all_images, all_labels), total=len(all_images))):
    resized_img = cv2.resize(img, target_size)
    normalized_img = resized_img / 255.0
    save_img = (normalized_img * 255).astype(np.uint8)

    class_dir = os.path.join(preprocessed_dir, label)
    os.makedirs(class_dir, exist_ok=True)
    img_path = os.path.join(class_dir, f"{i}.jpg")
    cv2.imwrite(img_path, save_img)
    processed_count += 1

print("\n✅ Preprocessing complete.")
print(f"📁 Saved preprocessed images to: {preprocessed_dir}")
print(f"📊 Total images processed: {processed_count}")
print(f"🔢 Classes found: {len(set(all_labels))}")
