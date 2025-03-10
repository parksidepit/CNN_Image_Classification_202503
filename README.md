# CNN Image Classification for Master Thesis

This project implements a complete pipeline for image classification using Convolutional Neural Networks (CNNs) based on
EfficientNet. The pipeline is designed for a master thesis and consists of several modules that handle data preparation,
training, evaluation, and a web interface for inference.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [Modules](#modules)
    - [Data Preparation](#data-preparation)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Duplicate Checker](#duplicate-checker)
    - [Web Application](#web-application)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project is developed as part of a master thesis and focuses on building and evaluating CNN models for classifying
images into SOC categories. The pipeline includes:

1. **Data Preparation:**
    - Reading an Excel file to extract metadata.
    - Renaming and sorting images into SOC-specific folders.
    - Splitting the sorted images into training, validation, and test sets.
    - Augmenting classes with insufficient data.

2. **Training:**
    - Training an EfficientNetB0 model with the last 70 layers unfrozen.
    - Using early stopping and model checkpointing.
    - Saving the trained model and training history.

3. **Evaluation:**
    - Evaluating the trained model on the test dataset.
    - Generating a classification report, confusion matrix, ROC curves, per-class accuracy bar chart, and
      misclassification heatmap.
    - Saving all evaluation results in a dedicated reports directory.

4. **Duplicate Checker:**
    - Checks for duplicate images across train/val/test splits and moves duplicates from the test set.

5. **Web Application:**
    - A Flask web app for inference that allows image uploads and displays the top-5 predictions.

Additional modules (e.g., hybrid CNN approach, training multiple backbones) are planned for future work.

## Project Structure

The project follows a modular structure:

```
CNN_Image_Classification_202503/
├── .venv/                      # Local Python virtual environment (ignored by .gitignore)
├── build/                      # Sphinx/LaTeX build outputs (ignored by .gitignore)
├── config/
│   └── config.yaml             # Central configuration file for paths, hyperparameters, etc.
├── data/                       # Main data folder; raw data is generally ignored
│   ├── classified_results/     # Images classified by the web app (grouped by predicted class)
│   ├── processed/              # SOC-based folders after Excel processing
│   ├── rti_results_export_neu/ # Original image subfolders ending with "_a" (ignored)
│   ├── splits/                 # Train/Val/Test directories created from processed data
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── uploads/                # Uploaded images via the web app (ignored if large)
│   ├── background.jpg          # Background image for the web interface
│   └── Sealings_workspace.xlsx # Excel file with metadata (ignored if large/private)
├── models/                     # Trained model (by default *not* ignored in .gitignore)
├── reports/                    # Evaluation outputs (plots, CSV, logs); can contain large files
├── source/                     # Sphinx source folder (ignored by .gitignore if desired)
├── src/
│   ├── data/
│   │   # (Optional) Additional data-related modules
│   ├── templates/             # HTML templates (upload.html, results.html) for the Flask web app
│   ├── __init__.py            # Empty file to recognize src/ as a Python package
│   ├── data_preparation.py    # Excel-based processing, augmentation, Train/Val/Test splitting
│   ├── duplicate_checker.py   # Detects duplicates in test vs. train/val and moves them
│   ├── evaluate.py            # Loads trained model, evaluates (ROC, confusion matrix, etc.)
│   ├── train.py               # Trains an EfficientNetB0 (last 70 layers unfrozen)
│   ├── train_multiple_backbones.py
│   │   # Compares multiple CNN backbones (VGG, ResNet, EfficientNet) using Hyperopt
│   └── webapp.py              # Flask-based upload and classification web application
├── .gitattributes              # Git attributes (e.g., text/binary settings)
├── .gitignore                  # Lists files/folders to ignore (e.g., .venv, data/, build/)
├── README.md                   # Main project documentation/overview
└── requirements.txt            # Python dependencies (TensorFlow, Flask, etc.)

```

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone <your_repository_url>
   cd CNN_Image_Classification_202503
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment:**

    - **Windows:**
      ```bash
      .venv\Scripts\activate
      ```
    - **macOS/Linux:**
      ```bash
      source .venv/bin/activate
      ```

4. **Install Dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Configuration

All configuration parameters are stored in `config/config.yaml`. Below is an example configuration:

```yaml
data:
  excel_path: data/Sealings_workspace.xlsx
  source_folder: data/rti_results_export_neu
  sorted_folder: data/processed
  splits:
    train: data/splits/train
    val: data/splits/val
    test: data/splits/test

training:
  image_size: [ 224, 224 ]
  batch_size: 16
  epochs: 15
  learning_rate: 5e-5
  dropout: 0.4
  dense_units: 512
  output_dir: models
  model_filename: optimized_efficientnet.keras

evaluation:
  output_dir: reports
```

**Note:** Adjust the paths and hyperparameters as needed.

## Modules

### Data Preparation

- **File:** `src/data_preparation.py`
- **Functionality:**
    - Processes an Excel file to extract SOC metadata and renames/copies images into SOC-based folders.
    - Splits the images into train, validation, and test sets.
    - Applies data augmentation if a class has fewer than the minimum required images.
- **Usage:**
  ```bash
  python src/data_preparation.py
  ```
  
### Duplicate Checker

- **File:** `src/duplicate_checker.py`
- **Functionality:**
    - Computes image hashes for images in train and validation sets.
    - Checks the test set for duplicates and moves any duplicates to a designated duplicates folder.
- **Usage:**
  ```bash
  python src/duplicate_checker.py
  ```

### Training Multiple Backbones  
- **File:** `src/train_multiple_backbones.py`
- **Functionality:**
    - Compares different CNN backbones (e.g., VGG16, ResNet50, EfficientNetB0) on the same dataset.
    - Uses Hyperopt for hyperparameter tuning (learning rate, dropout, etc.).
    - Saves model weights and evaluation results for each backbone.

### Training

- **File:** `src/train.py`
- **Functionality:**
    - Loads training parameters from the configuration.
    - Uses ImageDataGenerators to load data from train, validation, and test splits.
    - Builds an EfficientNetB0 model with the first 70 layers frozen and the remaining layers trainable.
    - Trains the model with early stopping and model checkpointing.
    - Saves the trained model and training history.
- **Usage:**
  ```bash
  python src/train.py
  ```

### Evaluation

- **File:** `src/evaluate.py`
- **Functionality:**
    - Loads the trained model and test data.
    - Computes evaluation metrics: classification report, confusion matrix, ROC curves, per-class accuracy bar chart,
      misclassification heatmap, and overall metrics.
    - Saves all evaluation outputs in the reports folder.
- **Usage:**
  ```bash
  python src/evaluate.py
  ```

### Web Application

- **File:** `src/webapp.py`
- **Functionality:**
    - Implements a Flask web application that loads the trained model.
    - Provides an interface to upload images and displays the top-5 predictions.
- **Usage:**
  ```bash
  python src/webapp.py
  ```
  After starting the application, open your browser and go to:

  ```bash
  http://127.0.0.1:5000
  ```


- **Usage:**
  ```bash
  python src/train_multiple_backbones.py
  ```
## Usage

1. **Prepare Data:**  
   Run the data preparation script:
   ```bash
   python src/data_preparation.py
   ```
   This will process your Excel file, rename images, and split them into train/val/test sets.

2. **Train the Model:**  
   Run the training script:
   ```bash
   python src/train.py
   ```
   The trained model and training history will be saved in the `models/` folder.

3. **Evaluate the Model:**  
   Run the evaluation script:
   ```bash
   python src/evaluate.py
   ```
   Evaluation reports and plots will be saved in the `reports/` folder.

4. **Run the Web Application:**  
   Run the web app:
   ```bash
   python src/webapp.py
   ```
   Then access the application through your browser to upload images and see predictions.

## Future Work

- **Hybrid Approach and Feature Extraction:**  
  Future updates may include additional modules for hybrid CNN approaches with feature extraction and t-SNE
  visualization.
- **Hyperparameter Tuning:**  
  More advanced tuning methods may be applied to improve model performance.

## Contributing

Feel free to fork this repository and submit pull requests for improvements. Please ensure that changes are well
documented and that the project structure is maintained.

## Contact
Peter Scheurer

BHT Berlin

Student Assistant

Vorderasiatisches Museum Berlin

## License

MIT License

Copyright (c) 2024 Peter Scheurer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.