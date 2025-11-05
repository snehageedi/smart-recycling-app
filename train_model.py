# train_model.py - UPGRADED with EfficientNetB0 and Advanced Augmentation

# ====================================================================
# PHASE 1: SETUP AND CONFIGURATION
# ====================================================================

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
# 🟢 UPDATED: Import EfficientNetB0 instead of VGG16
from tensorflow.keras.applications import EfficientNetB0 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# --- CONFIGURATION ---
DATASET_PATH = os.path.join(os.getcwd(), 'waste_dataset') 
IMAGE_SIZE = (224, 224) # Standard input size for EfficientNetB0
BATCH_SIZE = 32          
EPOCHS = 15              # Increased epochs slightly to account for the new architecture
# ---------------------

print("Starting Smart Waste Classifier Training Script (EfficientNetB0 Upgrade)...")

# ====================================================================
# PHASE 2: DATA PREPROCESSING AND LOADING
# ====================================================================

print("1. Preparing Data Generators...")

# 1. Image Data Generator for TRAINING (with Advanced Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,                # Normalize pixel values
    validation_split=0.2,          # Reserve 20% of the data for validation
    
    # 🟢 ADVANCED AUGMENTATION PARAMETERS:
    rotation_range=20,             # Rotate up to 20 degrees
    width_shift_range=0.1,         # Randomly shift image width
    height_shift_range=0.1,        # Randomly shift image height
    shear_range=0.15,              # Perspective transformation
    zoom_range=0.15,               # Zoom in/out
    horizontal_flip=True,          # Randomly flip horizontally
    brightness_range=[0.7, 1.0],   # Vary brightness (better robustness)
    fill_mode='nearest'
)

# 2. Image Data Generator for VALIDATION (NO AUGMENTATION, only rescaling)
# ❌ The validation generator MUST NOT have augmentation, only rescaling.
validation_datagen = ImageDataGenerator(
    rescale=1./255,                # Normalize pixel values
    validation_split=0.2          # Use the same 20% split reserved above
)


# Load the Training Data (80% of total)
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load the Validation Data (20% of total)
validation_generator = validation_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # Important: must point to the 'validation' subset
)

NUM_CLASSES = train_generator.num_classes
print(f"✅ Data found. Number of waste classes detected: {NUM_CLASSES}")

# ====================================================================
# PHASE 3: MODEL BUILDING (EFFICIENTNET TRANSFER LEARNING)
# ====================================================================

print("2. Building Model (EfficientNetB0 Transfer Learning)...")

# 1. Load the EfficientNetB0 base model
base_model = EfficientNetB0(
    input_shape=IMAGE_SIZE + (3,), 
    include_top=False,              # Exclude the original classification layer
    weights='imagenet'              # Use weights pre-trained on ImageNet
)

# Freeze the weights of the base model
base_model.trainable = False

# 2. Build the Final Architecture
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),        # 🟢 Recommended for EfficientNet: Replaces Flatten for better performance
    Dense(512, activation='relu'),   # Custom hidden layer
    Dropout(0.5),                    # Dropout to prevent overfitting
    Dense(NUM_CLASSES, activation='softmax') # Output layer
])

# 3. Compile the model
model.compile(
    optimizer='adam',                       
    loss='categorical_crossentropy',        
    metrics=['accuracy']                
)

print("\nModel Summary:")
model.summary()

# ====================================================================
# PHASE 4: TRAINING AND SAVING
# ====================================================================

print("\n3. Starting Model Training...")

# Train the model
# Added a simple callback for early stopping to prevent waste of time if validation accuracy plateaus
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)

# Save the trained model
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'smart_waste_classifier_model.h5')
model.save(MODEL_SAVE_PATH)
print(f"\nModel trained and saved successfully at: {MODEL_SAVE_PATH} 🎉")
