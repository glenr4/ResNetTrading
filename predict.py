import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('./model')

# Load and preprocess the test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the pixel values
x_test = x_test / 255.0

# One-hot encode the labels
y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test_categorical)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Make predictions
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test_categorical, axis=1)

# Generate classification report and confusion matrix
print('Classification Report')
print(classification_report(true_labels, predicted_labels))
print('Confusion matrix')
print(confusion_matrix(true_labels, predicted_labels))
