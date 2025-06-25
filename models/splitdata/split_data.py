import os
import shutil
import random

train_dir = r"dataset\train"
val_dir = r"dataset\val"
val_split = 0.2  # 20% for validation

# Get list of classes
classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

for cls in classes:
    # Create class directory in validation
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    # Get all images in class
    class_path = os.path.join(train_dir, cls)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]
    # Shuffle and split
    random.shuffle(images)
    val_size = int(len(images) * val_split)
    val_images = images[:val_size]
    # Move validation images
    for img in val_images:
        shutil.move(os.path.join(class_path, img), os.path.join(val_dir, cls, img))

print("Validation set created successfully!")