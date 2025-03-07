"""
duplicate_checker.py

This module checks for duplicate images across the training, validation, and test directories.
It computes an MD5 hash for each image in the train and validation splits and then checks the test split.
If a duplicate is found in the test set that already exists in train or val, the duplicate is moved
to a designated duplicates directory.
"""

import os
import hashlib
import shutil
import yaml

def get_project_root():
    """
    Returns the absolute path to the project root directory.
    This function calculates the directory one level above the current file location.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(current_dir, '..'))

def load_config():
    """
    Loads configuration settings from a YAML file located in the 'config' folder at the project root.

    Returns:
        dict: A dictionary containing configuration parameters.
    """
    project_root = get_project_root()
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def compute_image_hash(image_path):
    """
    Computes the MD5 hash for a given image file.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        str: The MD5 hash in hexadecimal format.
    """
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def remove_duplicates(train_dir, val_dir, test_dir, duplicate_dir):
    """
    Checks for duplicate images in the test set that already exist in the training or validation sets.
    If duplicates are found, they are moved to a duplicate directory.

    Parameters:
        train_dir (str): Path to the training images directory.
        val_dir (str): Path to the validation images directory.
        test_dir (str): Path to the test images directory.
        duplicate_dir (str): Path to the directory where duplicates will be moved.
    """
    # Collect hashes from train and validation sets
    train_val_hashes = {}
    for split_dir in [train_dir, val_dir]:
        for class_folder in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_folder)
            if os.path.isdir(class_path):
                for img in os.listdir(class_path):
                    img_path = os.path.join(class_path, img)
                    if os.path.isfile(img_path):
                        img_hash = compute_image_hash(img_path)
                        train_val_hashes[img_hash] = img_path

    duplicates_found = 0
    os.makedirs(duplicate_dir, exist_ok=True)

    # Check test set for duplicates
    for class_folder in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_folder)
        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                if os.path.isfile(img_path):
                    img_hash = compute_image_hash(img_path)
                    if img_hash in train_val_hashes:
                        duplicates_found += 1
                        print(f"Duplicate found in test set: {img_path}")
                        new_path = os.path.join(duplicate_dir, img)
                        shutil.move(img_path, new_path)

    print(f"Duplicate checking complete. {duplicates_found} duplicate images moved to {duplicate_dir}.")

if __name__ == "__main__":
    config = load_config()
    project_root = get_project_root()

    # Define split directories based on the configuration
    train_dir = os.path.join(project_root, config['data']['splits'].get('train', 'data/splits/train'))
    val_dir = os.path.join(project_root, config['data']['splits'].get('val', 'data/splits/val'))
    test_dir = os.path.join(project_root, config['data']['splits'].get('test', 'data/splits/test'))
    duplicate_dir = os.path.join(project_root, 'data/splits/duplicates_in_test')

    remove_duplicates(train_dir, val_dir, test_dir, duplicate_dir)
