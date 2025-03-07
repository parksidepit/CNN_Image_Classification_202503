"""
evaluate.py

This module evaluates a trained EfficientNetB0 model on the test split.
It loads the test data using ImageDataGenerator and computes several evaluation metrics:
- Overall test accuracy and a detailed classification report.
- A confusion matrix heatmap.
- ROC curves for each class.
- A per-class accuracy bar chart.
- A misclassification heatmap showing which classes are often confused.
All outputs are saved to a designated evaluation output directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from itertools import cycle
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

def evaluate_model():
    """
    Evaluates the trained model on the test dataset and generates several evaluation outputs:
    - Classification report (saved as a text file)
    - Confusion matrix heatmap (saved as PNG)
    - ROC curves for each class (saved as PNG)
    - Per-class accuracy bar chart (saved as PNG)
    - Misclassification heatmap (saved as PNG)
    - Overall metrics (saved as CSV)
    """
    config = load_config()
    project_root = get_project_root()

    # Build absolute paths using configuration
    test_rel_dir = config['data']['splits'].get('test', 'data/splits/test')
    test_dir = os.path.join(project_root, test_rel_dir)

    # Model path (from training section; defaults to models/optimized_efficientnet.keras)
    model_output_dir = config['training'].get('output_dir', 'models')
    model_path = os.path.join(project_root, model_output_dir, "optimized_efficientnet.keras")

    # Evaluation output directory: either from config or default to "reports"
    if 'evaluation' in config and 'output_dir' in config['evaluation']:
        eval_output_dir = os.path.join(project_root, config['evaluation']['output_dir'])
    else:
        eval_output_dir = os.path.join(project_root, 'reports')
    os.makedirs(eval_output_dir, exist_ok=True)

    # Get image dimensions and batch size from training config
    img_width, img_height = config['training'].get('image_size', [224, 224])
    batch_size = config['training'].get('batch_size', 16)

    # Set up the ImageDataGenerator for test data with EfficientNet preprocessing
    test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Load the trained model
    model = load_model(model_path)

    # Evaluate overall test performance
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Generate predictions and related metrics
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print("Classification Report:")
    print(report)
    report_path = os.path.join(eval_output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(eval_output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # ROC Curves for each class
    # Binarize labels for ROC calculation
    y_true_bin = label_binarize(y_true, classes=range(len(class_labels)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(12, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'purple', 'red', 'brown', 'olive',
                    'cyan', 'magenta', 'gold', 'navy', 'lime', 'teal', 'pink', 'gray', 'black', 'violet',
                    'indigo', 'coral', 'salmon', 'khaki', 'plum', 'orchid', 'sienna'])
    for i, color in zip(range(len(class_labels)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve for {class_labels[i]} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    roc_path = os.path.join(eval_output_dir, "roc_curves.png")
    plt.savefig(roc_path)
    plt.close()

    # Per-class accuracy: calculate and plot a bar chart
    per_class_accuracy = {}
    for cls in range(len(class_labels)):
        idx = np.where(y_true == cls)[0]
        correct = np.sum(y_pred[idx] == y_true[idx])
        per_class_accuracy[class_labels[cls]] = correct / len(idx) if len(idx) > 0 else 0

    plt.figure(figsize=(12, 6))
    acc_df = pd.DataFrame(list(per_class_accuracy.items()), columns=["Class", "Accuracy"])
    sns.barplot(x="Class", y="Accuracy", data=acc_df, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Per-Class Accuracy")
    plt.tight_layout()
    acc_path = os.path.join(eval_output_dir, "per_class_accuracy.png")
    plt.savefig(acc_path)
    plt.close()

    # Misclassification heatmap: create a pivot table of misclassifications
    misclassified = np.where(y_pred != y_true)[0]
    if len(misclassified) > 0:
        misclass_data = []
        for idx in misclassified:
            true_cls = class_labels[y_true[idx]]
            pred_cls = class_labels[y_pred[idx]]
            misclass_data.append((true_cls, pred_cls))
        misclass_df = pd.DataFrame(misclass_data, columns=["True Class", "Predicted Class"])
        pivot_table = misclass_df.pivot_table(index="True Class", columns="Predicted Class", aggfunc=len, fill_value=0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt="d", cmap="Reds")
        plt.title("Misclassification Heatmap")
        misclass_path = os.path.join(eval_output_dir, "misclassification_heatmap.png")
        plt.savefig(misclass_path)
        plt.close()
    else:
        print("No misclassifications to report.")
        misclass_path = None

    # Save overall metrics in a CSV file
    overall_metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    metrics_df = pd.DataFrame(list(overall_metrics.items()), columns=["Metric", "Value"])
    metrics_csv_path = os.path.join(eval_output_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    print("Evaluation complete. Reports and plots saved in:", eval_output_dir)

if __name__ == "__main__":
    evaluate_model()
