"""
webapp.py

This module implements a Flask web application for image classification using a trained model.
It loads the trained model from the models folder and provides an interface to upload an image,
which is then classified. The predictions (top 5) are displayed along with the uploaded image.

Folder structure assumptions:
- The project root contains a "models" folder with your trained model.
- The "src/templates" folder contains HTML templates (e.g., "upload.html" and "results.html").
- The "src/uploads" folder will store uploaded images, and "src/classified_results" will store copies of images classified by the predicted class.
"""

import os
import json
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


# Initialize Flask app
app = Flask(__name__)

# Load configuration and set paths
config = load_config()
project_root = get_project_root()

# Use the model filename "optimized_efficientnet.keras"
model_filename = "optimized_efficientnet.keras"
model_path = os.path.join(project_root, config['training'].get('output_dir', 'models'), model_filename)
model = load_model(model_path)

# Set upload and result directories (inside src folder)
UPLOAD_FOLDER = os.path.join(project_root, "src", "uploads")
RESULT_FOLDER = os.path.join(project_root, "src", "classified_results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# CLASS_LABELS ordered as in your PDF (29 classes)
CLASS_LABELS = {
    0: 'SOC1',
    1: 'SOC11',
    2: 'SOC12',
    3: 'SOC13',
    4: 'SOC15',
    5: 'SOC17',
    6: 'SOC19',
    7: 'SOC2',
    8: 'SOC20',
    9: 'SOC21',
    10: 'SOC22',
    11: 'SOC24',
    12: 'SOC25',
    13: 'SOC26',
    14: 'SOC27',
    15: 'SOC3',
    16: 'SOC30',
    17: 'SOC31b',
    18: 'SOC33',
    19: 'SOC37',
    20: 'SOC38',
    21: 'SOC4',
    22: 'SOC40b',
    23: 'SOC41',
    24: 'SOC5',
    25: 'SOC51',
    26: 'SOC52',
    27: 'SOC7',
    28: 'SOC8'
}


def predict_image(img_path):
    """
    Loads an image, preprocesses it, and predicts its class using the loaded model.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        list: A list of tuples (class_label, probability) for the top 5 predictions.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-5:][::-1]
    top_predictions = [(CLASS_LABELS[i], float(predictions[i])) for i in top_indices]
    return top_predictions


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return render_template("upload.html", message="No file selected.")
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Get predictions for the uploaded image
            predictions = predict_image(file_path)
            main_prediction = predictions[0][0]  # Top prediction

            # Optionally, copy the file into a folder corresponding to the predicted class
            class_folder = os.path.join(RESULT_FOLDER, main_prediction)
            os.makedirs(class_folder, exist_ok=True)
            shutil.copy(file_path, os.path.join(class_folder, filename))

            return render_template("results.html", predictions=predictions, image_url=f"/uploads/{filename}")
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
