import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for image manipulation

# Select which image in the test set to use (0-999)
image_index = 0  # Choose an index

# Load CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the saved model
model = tf.keras.models.load_model('./model')

# Load a single image from the test set (or load your own image)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)

image = x_test[image_index]
true_label = y_test[image_index][0]

# Preprocess the image
image_preprocessed = image/255 # Normalise
# This actually makes the prediction worse
# image_preprocessed = preprocess_input(image.copy()) # Normalisation, zero-centering, and scaling the pixel values
image_preprocessed = np.expand_dims(image_preprocessed, axis=0)  # Add batch dimension

# Make a prediction
predictions = model.predict(image_preprocessed)
predicted_label_index = np.argmax(predictions[0])
predicted_label = class_names[predicted_label_index]
probability = predictions[0][predicted_label_index]

# Display the image with label and probability
image_display = image.copy() #make copy for drawing on.
image_display = (image_display * 255).astype(np.uint8) # Convert back to uint8 and scale to 0-255
image_display = cv2.resize(image_display, (224,224)) # resize for text to fit

text = f"Predicted: {predicted_label} ({probability:.2f})"
true_text = f"True: {class_names[true_label]}"

# Add text to the image
cv2.putText(image_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(image_display, true_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Display
cv2.imshow('Prediction', image_display)
cv2.waitKey(0)
cv2.destroyAllWindows()