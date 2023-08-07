

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2

# Specify the path to the model file on your local machine
model_file_path = 'karim_model.h5'

# Load the model
model = tf.keras.models.load_model(model_file_path)

# Prepare test images directory
test_images_folder = 'C:/Users/abdel/Desktop/test'

# List of class labels (in the same order as the folders)
labels = ['back_complete','bad', 'Console', 'front_complete','side_left_complete','side_right_complete']
# Process and make predictions for each class folder
for label in labels:
    class_folder = os.path.join(test_images_folder, label)
    #if not os.path.exists(class_folder):
        #print(f"Class folder '{label}' not found.")
        #continue
    
    # Process each image in the class folder
    for filename in os.listdir(class_folder):
        img = cv2.imread(os.path.join(class_folder,filename))
        img = cv2.resize(img,(224, 224))
        #image_path = os.path.join(class_folder, filename)
        #img = load_img(image_path, target_size=(224, 224))  # Resize the image to match model input size
        img_array=np.array(img)
        #img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (required for model input)
        
        # Make predictions
        predictions = model.predict(img_array)
        
        # Assuming the model is a classifier with softmax activation, get the class with the highest probability
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = labels[predicted_class_index]
        
        print(f"Image: {filename}, Predicted Class: {predicted_class}")

