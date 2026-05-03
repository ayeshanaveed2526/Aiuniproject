import os
import shutil

base_path = r'c:\Users\PMLS\Desktop\AI assignment\dataset\flowers'
if not os.path.exists(base_path):
    print("Base path not found")
    exit()

for filename in os.listdir(base_path):
    if filename.endswith('.jpg'):
        # Extract class name from filename (e.g., 'bougainvillea' from 'bougainvillea_00003.jpg')
        class_name = '_'.join(filename.split('_')[:-1])
        if not class_name:
            continue
            
        class_dir = os.path.join(base_path, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
        shutil.move(os.path.join(base_path, filename), os.path.join(class_dir, filename))

print("Organization complete.")
