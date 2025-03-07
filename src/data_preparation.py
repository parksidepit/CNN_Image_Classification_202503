"""
data_preparation.py

This module implements a complete data preparation pipeline in two sequential steps:

1. Excel Processing and Image Renaming (Optional):
   - Reads an Excel file to extract metadata (museum and excavation numbers).
   - Creates a mapping from a combined identifier to a SOC number (digits extracted from a specified column).
   - Copies and renames images from the source folder to a "processed" folder using SOC-based subfolders.
   - Folders with no valid SOC number (or resulting in "NaN") are skipped.

2. Class Distribution Check, Augmentation, and Data Splitting:
   - Assumes that the processed folder now contains SOC subfolders (e.g., SOC1, SOC2, etc.).
   - Checks whether each SOC folder meets a minimum sample requirement; if not, images are augmented.
   - Splits images into training, validation, and test sets according to predefined ratios.
   - Copies the images into corresponding directories.
   - After splitting, each class folder in train, validation, and test is checked and augmented further if needed.

If the processed folder is already populated, the Excel processing step is skipped.
"""

import os
import shutil
import random
import numpy as np
import pandas as pd
import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

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

def process_excel_and_rename_images():
    """
    Processes an Excel file to extract metadata and rename/copy images accordingly.

    Steps:
      1. Load configuration and build absolute paths for the Excel file, source folder, and processed folder.
      2. Read the Excel file and strip extraneous spaces.
      3. Create a mapping between a combined identifier (museum_no + excavation_no) and a SOC number.
         Uses a regex that captures the first occurrence of digits in the SOC cell.
      4. Print the unique SOC numbers extracted for debugging.
      5. Iterate over folders in the source directory that contain '_a'. For each folder:
         - Extract museum and excavation numbers.
         - If no valid SOC number is found or the SOC number is invalid (e.g. "NaN"), skip the folder.
         - Otherwise, copy and rename images from that folder into a target subfolder labeled "SOC{soc_number}".
    """
    config = load_config()
    project_root = get_project_root()

    excel_rel_path = config['data'].get('excel_path', 'data/Sealings_workspace.xlsx')
    source_rel_folder = config['data'].get('source_folder', 'data/rti_results_export_neu')
    processed_rel_folder = config['data'].get('sorted_folder', 'data/processed')

    excel_path = os.path.join(project_root, excel_rel_path)
    source_folder = os.path.join(project_root, source_rel_folder)
    processed_folder = os.path.join(project_root, processed_rel_folder)

    # Read the Excel file and remove extraneous spaces from string values
    df = pd.read_excel(excel_path, dtype=str, header=None)
    df = df.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    museum_col = 2
    excavation_col = 4
    soc_col = 6

    if max(museum_col, excavation_col, soc_col) >= df.shape[1]:
        raise ValueError("The Excel file does not have enough columns to extract required values.")

    # Create mapping from combined identifier to SOC number.
    df["ID"] = df[museum_col].str.replace(" ", "") + "_" + df[excavation_col].str.replace(" ", "")
    id_to_soc = dict(zip(df["ID"], df[soc_col].str.replace(" ", "").str.extract(r'(\d+)')[0]))

    # Debug: print unique SOC numbers extracted
    valid_soc = [soc for soc in id_to_soc.values() if soc is not None and str(soc).lower() != "nan"]
    unique_soc = sorted(set(valid_soc))
    print("Unique SOC numbers extracted:", unique_soc)
    print("Count of unique SOC numbers:", len(unique_soc))

    os.makedirs(processed_folder, exist_ok=True)

    # Process each folder in the source directory
    for folder in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder)
        if "_a" in folder and os.path.isdir(folder_path):
            parts = folder.split("_")
            if len(parts) < 2:
                continue

            museum_no = parts[0]
            excavation_no = parts[1]
            combined_id = museum_no + "_" + excavation_no

            if combined_id not in id_to_soc:
                print(f"No SOC number found for {combined_id}, skipping...")
                continue

            soc_number = id_to_soc[combined_id]
            # Skip folders with invalid SOC number (including NaN)
            if soc_number is None or str(soc_number).lower() == "nan":
                print(f"Invalid SOC number for {combined_id}; skipping folder.")
                continue

            target_dir = os.path.join(processed_folder, f"SOC{soc_number}")
            target_dir = os.path.normpath(target_dir)
            os.makedirs(target_dir, exist_ok=True)

            for img in os.listdir(folder_path):
                if img.startswith("plane_") and img.endswith(".jpg"):
                    plane_no = img.split("_")[1]
                    new_filename = f"SOC{soc_number}_{museum_no}_{excavation_no}_a_plane_{plane_no}.jpg"
                    new_filepath = os.path.normpath(os.path.join(target_dir, new_filename))
                    old_filepath = os.path.join(folder_path, img)
                    shutil.copy(old_filepath, new_filepath)
                    print(f"Copied: {old_filepath} -> {new_filepath}")

    print("Excel processing and image renaming completed.")

def augment_images(class_dir, target_count, augmentation):
    """
    Augments images in a given class folder until the target count is reached.

    Parameters:
        class_dir (str): The full path of the class folder.
        target_count (int): The desired minimum number of images.
        augmentation (ImageDataGenerator): The augmentation generator.
    """
    # Get full paths of images in the folder
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith('.jpg')]
    current_count = len(images)
    while current_count < target_count:
        img_path = random.choice(images)
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        aug_img = augmentation.flow(img, batch_size=1)[0].astype('uint8')
        aug_img = array_to_img(aug_img[0])
        new_filename = f"aug_{current_count}.jpg"
        new_full_path = os.path.join(class_dir, new_filename)
        aug_img.save(new_full_path)
        # Append the full path to the list so that future iterations use it
        images.append(new_full_path)
        current_count += 1

def augment_in_split(split_class_folder, target_count, augmentation):
    """
    Augments images in a split class folder until the target count is reached.

    Parameters:
        split_class_folder (str): The full path of the class folder in the split directory.
        target_count (int): The desired minimum number of images.
        augmentation (ImageDataGenerator): The augmentation generator.
    """
    images = [os.path.join(split_class_folder, img) for img in os.listdir(split_class_folder) if img.endswith('.jpg')]
    current_count = len(images)
    while current_count < target_count:
        img_path = random.choice(images)
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        aug_img = augmentation.flow(img, batch_size=1)[0].astype('uint8')
        aug_img = array_to_img(aug_img[0])
        new_filename = f"aug_{current_count}.jpg"
        new_full_path = os.path.join(split_class_folder, new_filename)
        aug_img.save(new_full_path)
        images.append(new_full_path)
        current_count += 1

def sort_and_split_images():
    """
    Sorts images into training, validation, and test sets, augmenting underrepresented classes.

    Steps:
      1. Load configuration and build absolute paths for the processed folder and split directories.
      2. Clear and recreate the target split directories.
      3. Iterate over each SOC folder in the processed folder.
         Skip any folder with an invalid name (e.g., containing "nan").
      4. For each valid class, gather all image filenames.
      5. If the number of images is below the minimum threshold, augment images in the processed folder until reaching the threshold.
      6. Split the images into training, validation, and test sets based on predefined ratios.
      7. Copy the images to the corresponding directories.
      8. For each class folder in each split (train, val, test), check if there are at least the minimum required images;
         if not, augment within that split folder until the threshold is met.
    """
    config = load_config()
    project_root = get_project_root()

    processed_rel_folder = config['data'].get('sorted_folder', 'data/processed')
    train_rel_dir = config['data']['splits'].get('train', 'data/splits/train')
    val_rel_dir = config['data']['splits'].get('val', 'data/splits/val')
    test_rel_dir = config['data']['splits'].get('test', 'data/splits/test')

    processed_folder = os.path.join(project_root, processed_rel_folder)
    train_dir = os.path.join(project_root, train_rel_dir)
    val_dir = os.path.join(project_root, val_rel_dir)
    test_dir = os.path.join(project_root, test_rel_dir)

    # Clear and recreate target directories
    for directory in [train_dir, val_dir, test_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    # Define split ratios and minimum sample count per class
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15
    min_samples = 10

    # Create an ImageDataGenerator for augmentation (for both processed and split folders)
    augmentation = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Iterate over each SOC folder in the processed folder
    for class_name in os.listdir(processed_folder):
        # Skip folders that contain "nan" (invalid SOC)
        if "nan" in class_name.lower():
            print(f"Skipping folder {class_name} due to invalid SOC number.")
            continue

        class_path = os.path.join(processed_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        # Gather all image filenames (full paths) in the processed folder for the class
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith('.jpg')]
        random.shuffle(images)

        if len(images) < min_samples:
            print(f"Class {class_name} has only {len(images)} images in processed folder. Augmenting...")
            augment_images(class_path, min_samples, augmentation)
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith('.jpg')]

        num_train = int(len(images) * train_split)
        num_val = int(len(images) * val_split)
        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]

        # Create class subdirectories in each target split folder
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Copy images to the respective directories
        for img_path in train_images:
            shutil.copy(img_path, os.path.join(train_dir, class_name, os.path.basename(img_path)))
        for img_path in val_images:
            shutil.copy(img_path, os.path.join(val_dir, class_name, os.path.basename(img_path)))
        for img_path in test_images:
            shutil.copy(img_path, os.path.join(test_dir, class_name, os.path.basename(img_path)))

        # After copying, check each split folder for the class and augment if necessary
        for split_folder in [train_dir, val_dir, test_dir]:
            class_split_folder = os.path.join(split_folder, class_name)
            images_in_split = [os.path.join(class_split_folder, img) for img in os.listdir(class_split_folder) if img.endswith('.jpg')]
            if len(images_in_split) < min_samples:
                print(f"In {split_folder}, class {class_name} has only {len(images_in_split)} images. Augmenting in split folder...")
                augment_in_split(class_split_folder, min_samples, augmentation)

    print("Image sorting and splitting completed.")

if __name__ == "__main__":
    # Load configuration and determine the processed folder path
    config = load_config()
    project_root = get_project_root()
    processed_folder = os.path.join(project_root, config['data'].get('sorted_folder', 'data/processed'))

    # If the processed folder is empty, run the Excel processing step; otherwise, skip it.
    if not os.listdir(processed_folder):
        process_excel_and_rename_images()
    else:
        print("Processed folder is not empty. Skipping Excel renaming step.")

    # Proceed with sorting and splitting images
    sort_and_split_images()
