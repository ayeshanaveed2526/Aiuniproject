import os
import shutil
import random

src_dir = r'dataset/flowers'
dest_dir = r'dataset/flowers_processed'

# Target classes
target_classes = ['bougainvillea', 'daisies', 'tulip']

# Rebuild processed dir
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
os.makedirs(dest_dir)

# Create other dir
other_dir = os.path.join(dest_dir, 'other')
os.makedirs(other_dir)

total_other = 0

for folder in os.listdir(src_dir):
    src_folder = os.path.join(src_dir, folder)
    if not os.path.isdir(src_folder):
        continue

    if folder in target_classes:
        # Copy ALL target images
        dest_folder = os.path.join(dest_dir, folder)
        shutil.copytree(src_folder, dest_folder)
        count = len(os.listdir(dest_folder))
        print(f"[OK] Copied {count} images for target class: {folder}")
    else:
        # Copy ALL images from non-target classes into 'other'
        images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            src_img = os.path.join(src_folder, img)
            dest_img = os.path.join(other_dir, f"{folder}_{img}")
            shutil.copy2(src_img, dest_img)
        total_other += len(images)
        print(f"  -> Added {len(images)} images from '{folder}' to 'other'")

print(f"\n[OK] Total 'other' images: {total_other}")
print("[OK] Dataset fully prepared!")

# Print class counts
for cls in os.listdir(dest_dir):
    count = len(os.listdir(os.path.join(dest_dir, cls)))
    print(f"   {cls}: {count} images")
