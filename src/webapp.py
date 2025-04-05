"""
webapp.py

This module implements a Flask web application for image classification using a trained EfficientNet model.
It supports both single-file and folder uploads. Uploaded images are saved in the data/uploads folder,
and a copy of each classified image is saved in data/classified_results under the predicted class.
We also serve a background image from the data folder, and disclaimers are shown on both pages.
"""

import os
import shutil
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import yaml

def get_project_root():
    """
    Returns the absolute path to the project root directory.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(current_dir, '..'))

def load_config():
    """
    Loads configuration settings from a YAML file located in the 'config' folder at the project root.
    """
    project_root = get_project_root()
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

app = Flask(__name__)

# Route to serve files from the data folder
project_root = get_project_root()
@app.route('/data/<path:filename>')
def data_files(filename):
    return send_from_directory(os.path.join(project_root, "data"), filename)

# Load config
config = load_config()

# Ensure data folder exists
data_folder = os.path.join(project_root, "data")
os.makedirs(data_folder, exist_ok=True)

# Define subfolders for uploads and classification
UPLOAD_FOLDER = os.path.join(data_folder, "uploads")
RESULT_FOLDER = os.path.join(data_folder, "classified_results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model_filename = "optimized_efficientnet.keras"
model_path = os.path.join(
    project_root,
    config['training'].get('output_dir', 'models'),
    model_filename
)
model = load_model(model_path)

# Class labels (29)
CLASS_LABELS = {
    0: 'SOC1', 1: 'SOC11', 2: 'SOC12', 3: 'SOC13', 4: 'SOC15',
    5: 'SOC17', 6: 'SOC19', 7: 'SOC2', 8: 'SOC20', 9: 'SOC21',
    10: 'SOC22', 11: 'SOC24', 12: 'SOC25', 13: 'SOC26', 14: 'SOC27',
    15: 'SOC3', 16: 'SOC30', 17: 'SOC31b', 18: 'SOC33', 19: 'SOC37',
    20: 'SOC38', 21: 'SOC4', 22: 'SOC40b', 23: 'SOC41', 24: 'SOC5',
    25: 'SOC51', 26: 'SOC52', 27: 'SOC7', 28: 'SOC8'
}

def predict_image(img_path):
    """
    Loads an image, preprocesses it, and predicts its class using the loaded model.
    Returns a list of (class_label, probability) for the top-5 predictions.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    predictions = model.predict(arr)[0]
    top_indices = predictions.argsort()[-5:][::-1]
    return [(CLASS_LABELS[i], float(predictions[i])) for i in top_indices]

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves an uploaded file from the uploads folder."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/", methods=["GET", "POST"])
def upload_file_route():
    """
    Handles both single-file and folder uploads. On GET, shows upload.html.
    On POST, processes each file, obtains predictions, copies the file to
    classified_results, and returns results.html.
    """
    if request.method == "POST":
        files = request.files.getlist("file")
        print("Number of files received:", len(files))
        if not files:
            return render_template("upload.html", message="No file selected.")

        results = []
        for f in files:
            orig_filename = f.filename
            filename = secure_filename(os.path.basename(orig_filename))
            if not filename:
                filename = secure_filename(orig_filename)
            if not filename:
                continue

            file_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(file_path)

            top_5 = predict_image(file_path)
            main_class = top_5[0][0]

            # Copy the file into the predicted class folder
            class_folder = os.path.join(RESULT_FOLDER, main_class)
            os.makedirs(class_folder, exist_ok=True)
            shutil.copy(file_path, os.path.join(class_folder, filename))

            results.append({
                "image": filename,
                "image_url": f"/uploads/{filename}",
                "main_class": main_class,
                "top_5_predictions": top_5
            })

        if results:
            return render_template("results.html", results=results)
        else:
            return render_template("upload.html", message="No valid files found.")
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
