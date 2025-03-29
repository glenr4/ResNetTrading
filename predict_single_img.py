import tensorflow as tf
import numpy as np
import cv2
from tkinter import filedialog, Tk
import os

# Constants from train.py
IMG_SIZE = 224
CATEGORIES = ['down', 'neutral', 'up']

# Load the saved model
model = tf.keras.models.load_model('./model')

# Create Tk root
root = Tk()
root.withdraw()  # Hide the main window

# Open file dialog
file_path = filedialog.askopenfilename(
    title='Select trading chart image',
    filetypes=[('PNG files', '*.png')]
)

if file_path:
    # Load and preprocess the image
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess the image (matching train.py preprocessing)
    image_preprocessed = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_preprocessed = image_preprocessed.astype(np.float32) / 255.0
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)

    # Make prediction
    predictions = model.predict(image_preprocessed)
    predicted_label_index = np.argmax(predictions[0])
    predicted_label = CATEGORIES[predicted_label_index]
    probability = predictions[0][predicted_label_index]

    # Prepare display image
    image_display = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Add prediction text
    text = f"Prediction: {predicted_label} ({probability:.2f})"
    cv2.putText(image_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show probabilities for all categories
    y_pos = 60
    for idx, category in enumerate(CATEGORIES):
        prob_text = f"{category}: {predictions[0][idx]:.2f}"
        cv2.putText(image_display, prob_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y_pos += 30

    # Display
    cv2.imshow('Trading Prediction', image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No file selected")