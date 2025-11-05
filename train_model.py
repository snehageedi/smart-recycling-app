# train_model.py

# ====================================================================
# PHASE 1: SETUP AND CONFIGURATION
# ====================================================================

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# --- CONFIGURATION ---
# The script automatically looks for the 'waste_dataset' folder in the current directory (PBAS)
DATASET_PATH = os.path.join(os.getcwd(), 'waste_dataset') 
IMAGE_SIZE = (224, 224) # Standard input size for VGG16
BATCH_SIZE = 32          # Process 32 images at a time
EPOCHS = 10              # Number of full passes over the training data
# ---------------------

print("Starting Smart Waste Classifier Training Script...")

# ====================================================================
# PHASE 2: DATA PREPROCESSING AND LOADING
# ====================================================================

print("1. Preparing Data Generators...")

# 1. Image Data Generator for Training (with Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values from 0-255 to 0-1
    rotation_range=20,        # Rotate images up to 20 degrees
    width_shift_range=0.2,    # Randomly shift image width
    height_shift_range=0.2,   # Randomly shift image height
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2      # Reserve 20% of the data for validation
)

# Load the Training Data (80% of total)
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Use categorical for multiple classes (glass, plastic, etc.)
    subset='training'
)

# Load the Validation Data (20% of total)
validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

NUM_CLASSES = train_generator.num_classes
print(f"✅ Data found. Number of waste classes detected: {NUM_CLASSES}")

# ====================================================================
# PHASE 3: MODEL BUILDING (TRANSFER LEARNING)
# ====================================================================

print("2. Building Model (VGG16 Transfer Learning)...")

# 1. Load the VGG16 base model
base_model = tf.keras.applications.VGG16(
    input_shape=IMAGE_SIZE + (3,), # Input is 224x224 pixels with 3 color channels (RGB)
    include_top=False,             # Exclude VGG16's original classification layer
    weights='imagenet'             # Use weights pre-trained on the ImageNet dataset
)

# Freeze the weights of the base model (crucial for transfer learning)
base_model.trainable = False

# 2. Build the Final Architecture
model = Sequential([
    base_model,
    Flatten(),                     # Convert the 3D output of VGG16 into a 1D vector
    Dense(512, activation='relu'), # Custom hidden layer
    Dropout(0.5),                  # Dropout to prevent overfitting
    Dense(NUM_CLASSES, activation='softmax') # Output layer: one neuron per class
])

# 3. Compile the model
model.compile(
    optimizer='adam',                   # Optimizer
    loss='categorical_crossentropy',    # Loss function for multi-class classification
    metrics=['accuracy']                # Metric to track
)

print("\nModel Summary:")
model.summary()

# ====================================================================
# PHASE 4: TRAINING AND SAVING
# ====================================================================

print("\n3. Starting Model Training...")

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# Save the trained model
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'smart_waste_classifier_model.h5')
model.save(MODEL_SAVE_PATH)
print(f"\nModel trained and saved successfully at: {MODEL_SAVE_PATH} 🎉")