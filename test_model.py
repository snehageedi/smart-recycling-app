# test_model.py (MODIFIED FOR RECYCLABLE/NON-RECYCLABLE OUTPUT)

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURATION ---
IMAGE_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.getcwd(), 'smart_waste_classifier_model.h5')
DATASET_PATH = os.path.join(os.getcwd(), 'waste_dataset')

# ⚠️ Ensure this path is correct for your test image!
TEST_IMAGE_PATH = os.path.join(os.getcwd(), 'test_images', 'istockphoto-1791334240-612x612.jpg') 
# ---------------------

# === NEW FUNCTION: MAPS 6 CLASSES TO 2 CATEGORIES ===
def get_recyclability_status(waste_class):
    """Maps the detailed waste class to 'Recyclable' or 'Non-Recyclable'."""
    # Define which classes are Recyclable based on common standards and your dataset
    RECYCLABLE_CLASSES = ['plastic', 'glass', 'metal', 'paper'] 
    
    # Non-Recyclable includes 'organic', 'hazardous', and any other unknown classes
    if waste_class in RECYCLABLE_CLASSES:
        return "RECYCLABLE"
    else:
        return "NON-RECYCLABLE"
# ====================================================


def get_class_names(dataset_path, target_size):
    """Loads a temporary generator just to retrieve the class names."""
    try:
        test_datagen = image.ImageDataGenerator()
        temp_generator = test_datagen.flow_from_directory(
            dataset_path,
            target_size=target_size,
            class_mode='categorical',
            batch_size=1,
            shuffle=False
        )
        return list(temp_generator.class_indices.keys())
    except Exception as e:
        print(f"Error loading class names from dataset directory: {e}")
        print("Please ensure your 'waste_dataset' folder is structured correctly.")
        sys.exit(1)

def classify_waste_image(image_path, model, class_names):
    # 1. Load the image and resize
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)

    # 2. Prepare the image
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # 3. Make the prediction (Detailed 6-class prediction)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    # The detailed classification (e.g., 'plastic')
    detailed_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    
    # 4. Map the detailed class to the final recyclability status
    final_status = get_recyclability_status(detailed_class)

    # 5. Display the result
    plt.imshow(img)
    plt.title(f"Predicted Detailed Class: {detailed_class.upper()} ({confidence:.2f}%)\nFINAL STATUS: {final_status}")
    plt.axis('off')
    plt.show()

    return final_status

# --- Main Execution ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}.")
    print("Please ensure train_model.py ran successfully and created the .h5 file.")
    sys.exit(1)
    
if not os.path.exists(TEST_IMAGE_PATH):
    print(f"Error: Test image not found at {TEST_IMAGE_PATH}.")
    print("Please verify the filename in TEST_IMAGE_PATH is EXACTLY correct.")
    sys.exit(1)

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Retrieving class names from: {DATASET_PATH}")
CLASS_NAMES = get_class_names(DATASET_PATH, IMAGE_SIZE)

print("\n--- Starting Classification ---")
final_result = classify_waste_image(TEST_IMAGE_PATH, model, CLASS_NAMES)
print(f"Final Project Result: The item is considered: ♻️ {final_result}")
print("\n--- Starting Classification ---")
classification_result = classify_waste_image(TEST_IMAGE_PATH, model, CLASS_NAMES)
print(f"Final Classification Result: 🗑️ {classification_result.upper()}")