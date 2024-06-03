import os
import shutil
from random import shuffle

def split_data(source_folder, target_folder, split_ratios=(0.7, 0.15, 0.15)):
    """
    Splits the data from source_folder into train, validation, and test folders
    in the target_folder according to the specified ratios.
    """
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    classes = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]
    
    for cls in classes:
        # Create subdirectories in train, validation, test
        os.makedirs(os.path.join(target_folder, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(target_folder, 'validation', cls), exist_ok=True)
        os.makedirs(os.path.join(target_folder, 'test', cls), exist_ok=True)

        # List all files in the class directory
        all_files = os.listdir(os.path.join(source_folder, cls))
        all_files = [f for f in all_files if f.endswith('.png')]  # assuming files are .png, change if necessary
        shuffle(all_files)

        # Split files
        total_files = len(all_files)
        train_end = int(split_ratios[0] * total_files)
        validate_end = train_end + int(split_ratios[1] * total_files)

        train_files = all_files[:train_end]
        validate_files = all_files[train_end:validate_end]
        test_files = all_files[validate_end:]

        # Copy files to the respective directories
        for file in train_files:
            shutil.copy(os.path.join(source_folder, cls, file), os.path.join(target_folder, 'train', cls, file))
        for file in validate_files:
            shutil.copy(os.path.join(source_folder, cls, file), os.path.join(target_folder, 'validation', cls, file))
        for file in test_files:
            shutil.copy(os.path.join(source_folder, cls, file), os.path.join(target_folder, 'test', cls, file))

# Usage example
source_folder = 'data'  # Update this path
target_folder = 'datas'  # Update this path

split_data(source_folder, target_folder)
