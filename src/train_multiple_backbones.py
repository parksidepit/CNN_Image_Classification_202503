"""
train_multiple_backbones.py

This script trains multiple CNN backbones (VGG16, ResNet50, EfficientNetB0) on the same dataset,
using Hyperopt to search for optimal hyperparameters. It references config.yaml for paths.
Adapt it to your directory structure and config keys as needed.

Usage:
  python train_multiple_backbones.py

Dependencies:
  - tensorflow
  - keras
  - numpy
  - scikit-learn (for classification_report, confusion_matrix)
  - hyperopt
  - pyyaml
  - etc.

Make sure your config.yaml contains something like:

training:
  data_dir: "data/splits"       # or wherever your train/val/test are located
  output_dir: "models"          # where to save models
  ...

"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


def load_config():
    """
    Loads configuration settings from config/config.yaml relative to this file.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_preprocessing(backbone_name):
    """
    Returns the appropriate preprocessing function for the given backbone.
    """
    if backbone_name == 'VGG16':
        return vgg16_preprocess
    elif backbone_name == 'ResNet50':
        return resnet50_preprocess
    elif backbone_name == 'EfficientNetB0':
        return effnet_preprocess
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


def create_model(backbone_name, input_shape, num_classes, dropout, dense_units):
    """
    Builds a model with the specified backbone, plus a custom classifier head.
    """
    if backbone_name == 'VGG16':
        base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'ResNet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'EfficientNetB0':
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Freeze the base model
    base.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout),
        Dense(num_classes, activation='softmax')
    ])
    return model


def main():
    config = load_config()
    # Directories from config
    data_dir = config['training'].get('data_dir', 'data/splits')
    output_dir = config['training'].get('output_dir', 'models')
    os.makedirs(output_dir, exist_ok=True)

    # Setup paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')
    test_dir  = os.path.join(data_dir, 'test')

    # Some default params
    img_width, img_height = 224, 224
    batch_size = 16
    epochs = 10  # can adjust as needed

    # Check that directories exist
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    # We'll define a function that uses hyperopt to evaluate a single param set
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

    # We'll define a search space for learning_rate, dropout, dense_units
    search_space = {
        'learning_rate': hp.loguniform('learning_rate', -5, -2),  # ~ 1e-5 to 1e-2
        'dropout': hp.uniform('dropout', 0.3, 0.7),
        'dense_units': hp.choice('dense_units', [128, 256, 512])
    }

    # We'll loop over multiple backbones
    backbones = ['VGG16', 'ResNet50', 'EfficientNetB0']
    results = {}

    # Let's define a helper function for the objective
    def objective(params, backbone, train_gen, val_gen):
        try:
            model = create_model(
                backbone_name=backbone,
                input_shape=(img_width, img_height, 3),
                num_classes=len(train_gen.class_indices),
                dropout=params['dropout'],
                dense_units=params['dense_units']
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            history = model.fit(
                train_gen,
                steps_per_epoch=train_gen.samples // batch_size,
                validation_data=val_gen,
                validation_steps=val_gen.samples // batch_size,
                epochs=5,  # short training for hyperopt
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
                verbose=0
            )
            val_loss = min(history.history['val_loss'])
            return {'loss': val_loss, 'status': STATUS_OK}
        except Exception as e:
            print(f"Error in objective: {e}")
            return {'loss': float('inf'), 'status': STATUS_OK}

    # For each backbone, we'll do a hyperopt search, then train a final model with best params
    for backbone in backbones:
        print(f"\n=== Training with {backbone} ===")
        # Preprocessing function
        preprocess_fn = get_preprocessing(backbone)

        # Data augmentation
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            rotation_range=50,
            width_shift_range=0.3,
            height_shift_range=0.3,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.3,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical'
        )
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical'
        )
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Ensure class counts match
        if (len(train_gen.class_indices) != len(val_gen.class_indices) or
                len(train_gen.class_indices) != len(test_gen.class_indices)):
            raise ValueError("Number of classes differs between train/val/test directories.")

        # Hyperopt objective that references partial closure
        def local_objective(hp_params):
            return objective(hp_params, backbone, train_gen, val_gen)

        trials = Trials()
        best = fmin(
            fn=local_objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials
        )
        print(f"Best hyperparams for {backbone}: {best}")

        # Now let's train final model with best params
        final_lr = best['learning_rate']
        final_dropout = best['dropout']
        # best['dense_units'] is an index in [0,1,2], so we map it
        dense_list = [128, 256, 512]
        final_dense_units = dense_list[best['dense_units']]

        model = create_model(
            backbone_name=backbone,
            input_shape=(img_width, img_height, 3),
            num_classes=len(train_gen.class_indices),
            dropout=final_dropout,
            dense_units=final_dense_units
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=final_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model_name = f"model_{backbone}.h5"
        checkpoint_path = os.path.join(output_dir, model_name)
        checkpoint_cb = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        final_history = model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // batch_size,
            validation_data=val_gen,
            validation_steps=val_gen.samples // batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                       checkpoint_cb]
        )

        # Evaluate on test
        test_loss, test_acc = model.evaluate(test_gen, steps=test_gen.samples // batch_size)
        print(f"Test Accuracy for {backbone}: {test_acc*100:.2f}%")

        # Predictions
        y_pred = model.predict(test_gen, steps=test_gen.samples // batch_size)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_gen.classes[:len(y_pred_classes)]

        # Classification report
        labels = list(test_gen.class_indices.values())
        target_names = list(test_gen.class_indices.keys())
        report = classification_report(
            y_true, y_pred_classes,
            labels=labels,
            target_names=target_names
        )
        print(f"\nClassification Report for {backbone}:")
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f"Confusion Matrix for {backbone}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{backbone}.png"))
        plt.close()

        # Save final results
        results[backbone] = {
            'best_params': best,
            'test_accuracy': test_acc,
            'classification_report': report
        }

    print("\n=== Training completed for all backbones ===")
    # Optionally print summary results
    for backbone, data in results.items():
        print(f"\nBackbone: {backbone}")
        print(f"Best hyperparams: {data['best_params']}")
        print(f"Test accuracy: {data['test_accuracy']*100:.2f}%")
        print(f"Report:\n{data['classification_report']}")

if __name__ == "__main__":
    main()
