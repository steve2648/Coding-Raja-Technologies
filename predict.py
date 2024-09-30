import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Load the trained model
model = tf.keras.models.load_model('food_classification_model.h5')

# Define image size (same as the size used during training)
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Path to class labels (you can get this from the train data generator)
class_labels = list(train_data_gen.class_indices.keys())  # Change this as needed

# Function to preprocess a single image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size
    img_array /= 255.  # Rescale pixel values like during training
    return img_array

# Function to predict the class of a food image
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])  # Get the index of the highest predicted class
    predicted_class = class_labels[predicted_class_index]
    
    # Display the image with prediction
    plt.imshow(image.load_img(img_path))
    plt.title(f'Predicted: {predicted_class}')
    plt.show()

    return predicted_class

# Path to the new food image to classify
image_path = '/content/DATASET/DATASET/Train/cheesecake/1004807.jpg'

# Predict the class of the image
predicted_class = predict_image(image_path)
print(f'The predicted class for the image is: {predicted_class}')