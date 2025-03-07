"""
train.py

This module trains an EfficientNetB0 model on the prepared image data.
It loads training parameters from the configuration file and uses ImageDataGenerators
to load data from the train, validation, and test splits. The model is trained with
early stopping and model checkpointing, and the best model is saved to the output directory.
Training history is also saved as a CSV file.

This version unfreezes the last 70 layers of the EfficientNetB0 base model.
Usage:
    Run this script after running data_preparation.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

def build_model(input_shape, num_classes, dropout_rate, dense_units, learning_rate):
    """
    Builds and compiles an EfficientNetB0 model for classification.

    This function loads the EfficientNetB0 base with ImageNet weights and then freezes the first 70 layers.
    The remaining layers are made trainable. A GlobalAveragePooling2D layer, a Dense layer with a ReLU activation,
    a Dropout layer, and a final Dense output layer with softmax activation are then added.

    Parameters:
        input_shape (tuple): The input shape of the images.
        num_classes (int): Number of classes in the dataset.
        dropout_rate (float): Dropout rate to apply after the dense layer.
        dense_units (int): Number of units in the dense layer before the output.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        A compiled Keras model.
    """
    # Ensure learning_rate is a float
    learning_rate = float(learning_rate)

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the first 70 layers and unfreeze the rest
    for layer in base_model.layers[:70]:
        layer.trainable = False
    for layer in base_model.layers[70:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    """
    Trains an EfficientNetB0 model on the prepared dataset.

    The function:
      1. Loads training parameters and data directories from the configuration.
      2. Sets up ImageDataGenerators for training, validation, and test splits.
      3. Builds the model with the EfficientNetB0 base (last 70 layers trainable).
      4. Trains the model with early stopping and model checkpoint callbacks.
      5. Evaluates the model on the test set and saves training history.

    Returns:
        None
    """
    config = load_config()
    project_root = get_project_root()

    # Directories for data splits
    train_dir = os.path.join(project_root, config['data']['splits'].get('train', 'data/splits/train'))
    val_dir = os.path.join(project_root, config['data']['splits'].get('val', 'data/splits/val'))
    test_dir = os.path.join(project_root, config['data']['splits'].get('test', 'data/splits/test'))

    # Training parameters
    img_width, img_height = config['training'].get('image_size', [224, 224])
    batch_size = config['training'].get('batch_size', 16)
    epochs = config['training'].get('epochs', 15)
    learning_rate = config['training'].get('learning_rate', 5e-5)
    dropout_rate = config['training'].get('dropout', 0.4)
    dense_units = config['training'].get('dense_units', 512)

    # Set up ImageDataGenerators with EfficientNet preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    num_classes = len(train_generator.class_indices)
    print("Number of classes:", num_classes)
    print("Class indices:", train_generator.class_indices)

    # Build and compile the model (with last 70 layers trainable)
    model = build_model(input_shape=(img_width, img_height, 3),
                        num_classes=num_classes,
                        dropout_rate=dropout_rate,
                        dense_units=dense_units,
                        learning_rate=learning_rate)

    # Set up output directory and model checkpoint
    output_dir = os.path.join(project_root, config['training'].get('output_dir', 'models'))
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "optimized_efficientnet.keras")

    checkpoint_callback = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint_callback]
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Save training history as CSV
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")

if __name__ == "__main__":
    train_model()
