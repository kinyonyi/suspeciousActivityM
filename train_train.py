import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys

# Configuration 
CONFIG = {
    "data_path": os.path.join("dataset"),
    "img_size": (224, 224),
    "batch_size": 32,
    "val_split": 0.2,
    "epochs": 15,
    "learning_rate": 1e-3,
    "augmentation": True,
    "use_class_weights": True,
    "model_save_path": "suspicious_activity_model.h5"
}

# Dataset Validation 
def validate_dataset_path(config):
    """Verify dataset exists and has correct structure"""
    if not os.path.exists(config["data_path"]):
        print(f"\nError: Dataset folder not found at: {os.path.abspath(config['data_path'])}")
        print("Please create a 'dataset' folder with class subfolders")
        sys.exit(1)
    
    class_folders = [f for f in os.listdir(config["data_path"]) 
                   if os.path.isdir(os.path.join(config["data_path"], f))]
    
    if not class_folders:
        print(f"\nError: No class folders found in {config['data_path']}")
        print("Dataset structure should be: dataset/class1/, dataset/class2/, etc.")
        sys.exit(1)
    
    print(f"\nFound dataset at: {os.path.abspath(config['data_path'])}")
    print(f"Detected {len(class_folders)} classes: {class_folders}")

# Data Preparation
def prepare_data(config):
    validate_dataset_path(config)  # dataset check first before anything else
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=config["val_split"]
    )

    train_data = train_datagen.flow_from_directory(
        config["data_path"],
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode='categorical',
        subset='training'
    )

    val_data = train_datagen.flow_from_directory(
        config["data_path"],
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode='categorical',
        subset='validation'
    )

    print(f"\nTraining samples: {train_data.samples}")
    print(f"Validation samples: {val_data.samples}")
    
    return train_data, val_data

# Main Execution for training of the data
if __name__ == "__main__":
    print("\nChecking dataset...")
    train_data, val_data = prepare_data(CONFIG)
    
    # Rest of model code...
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*CONFIG["img_size"], 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(train_data.num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    print("\nStarting training...")
    model.fit(train_data,  validation_data=val_data, epochs=CONFIG["epochs"])
    
    model.save(CONFIG["model_save_path"])
    print(f"\nModel saved to {CONFIG['model_save_path']}")