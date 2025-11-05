# app.py - ULTIMATE Professional Smart Waste Classifier (Model Reassembly Enabled)

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
from PIL import Image
import time # 🟢 NEW: Needed for unique timestamps in feedback loop

# --- NEW IMPORTS for Model Reassembly ---
import io # Needed for creating an in-memory file object (BytesIO)
import glob # Needed for finding all model part files
# ----------------------------------------

# --- CONFIGURATION (UPDATED for model parts) ---
IMAGE_SIZE = (224, 224)

# We define the prefix used by the split_model.py script
MODEL_PART_PREFIX = 'model_part_' 

# NOTE: MODEL_PATH is no longer used for loading, but kept for reference
MODEL_PATH = os.path.join(os.getcwd(), 'smart_waste_classifier_model.h5') 
DATASET_PATH = os.path.join(os.getcwd(), 'waste_dataset')

# Using the validation accuracy you reported earlier (~63.33%)
MODEL_EXPECTED_ACCURACY = "63.33%"
# -----------------------------------------------------

# --- SESSION STATE INITIALIZATION ---
# 0: Input (Upload/Camera), 1: Confirm, 2: Results
if 'step' not in st.session_state:
    st.session_state['step'] = 0 
if 'img_source' not in st.session_state:
    st.session_state['img_source'] = None
if 'results' not in st.session_state:
    st.session_state['results'] = None

# === PROJECT LOGIC FUNCTIONS (Cached for Performance) ===

@st.cache_resource
def load_and_cache_model():
    """
    Reads model parts, combines them into a byte stream (BytesIO), 
    writes the stream to a temporary file, loads the Keras model from the path,
    and then deletes the temporary file.
    """
    st.info("Searching for model parts...")
    
    # 1. Locate all model parts (e.g., model_part_00, model_part_01, ...)
    model_parts = sorted(glob.glob(f'{MODEL_PART_PREFIX}*')) 
    
    if not model_parts:
        st.error(f"Error: Could not find any model files starting with '{MODEL_PART_PREFIX}' in the deployment directory. Did you upload all 11 parts?")
        st.stop()
        
    st.success(f"Found {len(model_parts)} model parts. Reassembling model now...")
    
    combined_bytes = bytearray()
    
    # 2. Read each part and append its content to the bytearray
    for part_file in model_parts:
        try:
            with open(part_file, 'rb') as f:
                combined_bytes.extend(f.read())
        except Exception as e:
            st.error(f"Failed to read model part {part_file}. Reassembly failed: {e}")
            st.stop()
            
    # 3. Create a BytesIO object (in-memory file) from the combined byte stream
    combined_stream = io.BytesIO(combined_bytes)
    
    # 4. Keras requires a file path (string) for load_model, not BytesIO.
    # We must write the combined stream to a temporary file path first.
    try:
        # Create a temporary file path in the current directory (Streamlit server storage)
        temp_model_path = "temp_combined_model.h5"
        
        # Write the combined stream to the temporary file
        with open(temp_model_path, 'wb') as f:
            f.write(combined_stream.getvalue())

        # Load the model from the temporary file path (which load_model requires)
        model = tf.keras.models.load_model(temp_model_path)
        
        # Clean up the temporary file immediately after loading to save space
        os.remove(temp_model_path)
        
        st.success("AI Model loaded successfully! Ready to classify.")
        return model
        
    except Exception as e:
        st.error(f"FATAL ERROR: Failed to load model from combined stream. Check TensorFlow/Keras version in requirements.txt. Error: {e}")
        st.stop()


@st.cache_resource
def get_class_names(dataset_path):
    """Retrieve class names (subfolder names) from the dataset directory."""
    try:
        # NOTE: This function still uses DATASET_PATH, ensure your 'waste_dataset' folder is uploaded if needed
        test_datagen = image.ImageDataGenerator()
        temp_generator = test_datagen.flow_from_directory(
            dataset_path,
            target_size=IMAGE_SIZE,
            class_mode='categorical',
            batch_size=1,
            shuffle=False
        )
        return list(temp_generator.class_indices.keys())
    except Exception as e:
        st.error(f"Error reading dataset structure: {e}")
        st.stop()

def get_recyclability_status(waste_class):
    """Maps the detailed waste class to 'RECYCLABLE' or 'NON-RECYCLABLE'."""
    RECYCLABLE_CLASSES = ['plastic', 'glass', 'metal', 'paper', 'cardboard']
    
    if waste_class in RECYCLABLE_CLASSES:
        return "RECYCLABLE"
    else:
        return "NON-RECYCLABLE"

def get_recycling_tip(waste_class):
    """Provides practical advice based on the predicted class."""
    TIPS = {
        'plastic': "Tip: Always rinse plastic containers before placing them in the recycling bin. They must be clean!",
        'glass': "Tip: Glass jars and bottles are 100% recyclable. Remove lids, but labels can stay.",
        'metal': "Tip: Aluminum cans and steel food cans are highly valuable. Crush them to save space!",
        'paper': "Tip: Paper, magazines, and envelopes are recyclable. However, shredded paper should be bagged.",
        'cardboard': "Tip: Cardboard boxes must be broken down flat before recycling. Remove all tape and packing materials.",
        'trash': "Tip: This is a non-recyclable item. Please place it in the general waste bin.",
        'hazardous': "Tip: DO NOT place in the regular trash. Take this item to a special hazardous waste facility.",
        'organic': "Tip: Organic waste (food scraps) is best suited for composting, not the recycling bin."
    }
    return TIPS.get(waste_class, "No specific tip available. Always check local guidelines.")


def classify_image_and_save_results(img, model, class_names):
    """Preprocess image, make prediction, and return a results dictionary."""
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    
    all_predictions = predictions[0] * 100
    
    predicted_class_index = np.argmax(all_predictions)
    
    detailed_class = class_names[predicted_class_index]
    confidence = all_predictions[predicted_class_index]
    
    final_status = get_recyclability_status(detailed_class)
    
    return {
        'detailed_class': detailed_class,
        'confidence': confidence,
        'final_status': final_status,
        'all_predictions': all_predictions
    }
    
# 🟢 NEW FUNCTION: For the User Feedback Loop
def save_misclassified_image(img, correct_label):
    """Saves the image to a 'user_feedback' folder for future retraining."""
    feedback_dir = 'user_feedback'
    if not os.path.exists(feedback_dir):
        # In a real deployed app, Streamlit may not allow writing to disk.
        # This is a placeholder; you'd replace this with saving to S3 or a database.
        st.warning("Cannot write feedback to disk in this environment. Skipping save.")
        return
        
    # Save the image with a unique name based on the corrected label
    filename = f"{feedback_dir}/{correct_label}_{int(time.time())}.png"
    img.save(filename)
    st.success(f"Feedback recorded! Thank you for improving the model.")


# --- NAVIGATION FUNCTIONS ---
def go_to_step(target_step):
    st.session_state['step'] = target_step

def reset_app():
    st.session_state['step'] = 0
    st.session_state['img_source'] = None
    st.session_state['results'] = None

# ====================================================================
# STREAMLIT UI CODE
# ====================================================================

# --- 1. Set Page Configuration and Custom Light Theme ---
st.set_page_config(
    page_title="AI Waste Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS for a modern light theme (kept from last version)
st.markdown(
    """
    <style>
    /* Main app background: Light Gray for elegance */
    .stApp {
        background-color: #f0f2f6;
        color: #1c2a39;
    }
    /* Headers and text */
    h1, h2, h3, h4, .stMarkdown {
        color: #1c2a39;
    }
    /* BUTTON STYLING: White button, Black Text (as requested) */
    .stButton>button {
        color: #1c2a39 !important; /* Black text */
        background-color: white !important; /* White button background */
        border: 2px solid #555555; /* Subtle dark border */
        border-radius: 10px; /* More rounded corners */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        transition: all 0.2s;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #e0e0e0 !important; /* Light hover effect */
        border-color: #000000;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    /* Container/Metric Boxes: Clean white background */
    [data-testid="stContainer"], [data-testid="stMetric"], [data-testid="stAlert"] {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    /* Success/Error boxes (using modern colors) */
    [data-testid="stSuccess"] { 
        border-color: #198754; 
        background-color: #d1e7dd; 
        color: #198754; 
    }
    [data-testid="stError"] { 
        border-color: #dc3545; 
        background-color: #f8d7da; 
        color: #dc3545;
    }
    [data-testid="stInfo"] { 
        border-color: #0d6efd; 
        background-color: #cfe2ff; 
        color: #0d6efd;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
        background-color: #e0e0e0; 
        color: #1c2a39;
        border-radius: 8px 8px 0 0;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.title("♻️ AI Smart Waste Classifier")
st.markdown("---")


# Load model outside the main block for efficiency
try:
    model = load_and_cache_model() 
    CLASS_NAMES = get_class_names(DATASET_PATH)
except:
    st.stop()


# --- WORKFLOW LOGIC ---

# --- Current Step Indicator (Optional, but helps user know where they are) ---
current_flow = {
    0: "Input / Capture Image",
    1: "Confirmation / Action",
    2: "Classification Report"
}.get(st.session_state['step'], "Start")

st.markdown(f"#### Current Workflow: **{current_flow}**")
st.markdown("---")


# --- STEP 0: INPUT/CAPTURE ---
if st.session_state['step'] == 0:
    st.markdown("### Upload or Capture Waste Item Image")
    
    # Metrics and Info 
    col_metrics, col_info = st.columns([1, 2])
    with col_metrics:
        st.markdown("#### Model Performance")
        # 🟢 UI FIX: Updated VGG16 to EfficientNetB0
        st.metric(label="Validation Accuracy", value=MODEL_EXPECTED_ACCURACY, delta="Deep Learning (EfficientNetB0)")
        st.caption("Expected reliability on unseen data.")
        
    with col_info:
        st.markdown("#### **Input Method**")
        st.markdown("""
        Use the tabs below to provide an image for classification.
        """)

    # Input Tabs
    tab_upload, tab_camera = st.tabs(["📁 UPLOAD IMAGE", "📸 USE LIVE CAMERA"])

    uploaded_file = None
    camera_input = None
    
    with tab_upload:
        uploaded_file = st.file_uploader("Upload a JPG, JPEG, or PNG image of the waste item:", type=["jpg", "jpeg", "png"])
        
    with tab_camera:
        camera_input = st.camera_input("Take a picture of the waste item now:")
    
    # Check for image submission
    if uploaded_file or camera_input:
        img_data = uploaded_file if uploaded_file else camera_input
        st.session_state['img_source'] = Image.open(img_data)
        go_to_step(1)
        st.rerun()

# --- STEP 1: CONFIRMATION & ACTION ---
elif st.session_state['step'] == 1:
    st.markdown("### Confirm Image and Classify")
    
    if st.session_state['img_source']:
        col_img, col_actions = st.columns([1, 1])

        with col_img:
            st.subheader("Image Ready")
            st.image(st.session_state['img_source'], use_column_width=True, caption="Review the image before classification.")
        
        with col_actions:
            st.subheader("Actions")
            # 🟢 UI FIX: Updated VGG16 to EfficientNetB0
            st.markdown("Click the button below to initiate the Deep Learning analysis. The image will be processed by the EfficientNetB0 model.")
            
            # Button to trigger classification and move to Step 2
            if st.button('🚀 CLASSIFY WASTE ITEM', use_container_width=True):
                # Run classification and store results
                # 🟢 UI FIX: Updated VGG16 to EfficientNetB0
                with st.spinner('Running EfficientNetB0 Transfer Learning Analysis...'):
                    results = classify_image_and_save_results(st.session_state['img_source'], model, CLASS_NAMES)
                
                st.session_state['results'] = results
                go_to_step(2)
                st.rerun()

            st.markdown("---")
            # Button to go back to Step 0
            if st.button('⬅️ Change Image', use_container_width=True):
                reset_app()
                st.rerun()

# --- STEP 2: RESULTS DISPLAY ---
elif st.session_state['step'] == 2:
    st.markdown("### Final Classification Report")
    results = st.session_state['results']
    img = st.session_state['img_source']
    
    if results and img:
        
        # Display image and results side-by-side
        col_img, col_result = st.columns([1, 1])
        
        with col_img:
            st.subheader("Analyzed Item")
            st.image(img, use_column_width=True)
            
        with col_result:
            st.subheader("Classification Summary")

            # --- Final Verdict ---
            st.markdown("#### **Final Verdict:**")
            if results['final_status'] == "RECYCLABLE":
                st.success(f"## ♻️ {results['final_status']}")
            else:
                st.error(f"## ❌ {results['final_status']}")
                
            st.markdown("---")
            
            # --- FEATURE 1: Confidence Visualization ---
            st.markdown(f"**Detailed Prediction:** **{results['detailed_class'].upper()}**")
            
            data = {'Confidence (%)': results['all_predictions']}
            df = pd.DataFrame(data, index=CLASS_NAMES)
            st.bar_chart(df, height=150, color="#198754") 
            
            st.caption(f"Model Confidence in {results['detailed_class'].upper()}: {results['confidence']:.2f}%")
            
            if results['confidence'] < 70.0:
                st.warning("⚠️ Low Confidence: Prediction is uncertain.")
            else:
                st.info("✅ High Confidence: Result is reliable.")
                
            # 🟢 NEW FEATURE: User Feedback Loop
            st.markdown("---")
            st.markdown("#### **Was this classification correct?**")
            
            col_yes, col_no = st.columns(2)
            
            if col_yes.button('✅ Yes, it was correct!', use_container_width=True):
                st.success("Great! Thank you for the confirmation.")
                
            if col_no.button('❌ No, it was wrong.', use_container_width=True):
                # Placeholder for the future feature to capture user correction
                st.session_state['feedback_mode'] = True
                
            # Logic to handle user correction if the previous button was clicked
            if 'feedback_mode' in st.session_state and st.session_state['feedback_mode']:
                st.warning(f"Please select the correct category for the item classified as **{results['detailed_class'].upper()}**:")
                
                # Create a list of classes excluding the current prediction
                other_classes = [c for c in CLASS_NAMES if c != results['detailed_class']]
                
                correct_class = st.selectbox("Select the correct waste category:", other_classes)
                
                if st.button("Submit Correction and Save Image"):
                    # Pass the original image and the correct class to the saving function
                    save_misclassified_image(img, correct_class)
                    del st.session_state['feedback_mode']
                    st.rerun()


        # --- Sorting Guidance Tip (Full Width Section) ---
        st.markdown("---")
        st.markdown("#### **Sorting Guidance**")
        tip = get_recycling_tip(results['detailed_class'])
        st.info(tip)
        
        st.markdown("---")
        # Button to restart the process
        if st.button('🔄 Start New Classification', use_container_width=True):
            reset_app()
            st.rerun()

# --- 4. Footer ---
st.markdown("---")
