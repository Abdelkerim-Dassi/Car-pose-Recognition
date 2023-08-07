from typing import Union

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2
import uuid

from fastapi import FastAPI, File, UploadFile

IMAGEDIR = "images/"
MODEL_FILE_PATH = "karim_model.h5"
LABELS = ['back_complete', 'bad', 'Console', 'front_complete', 'side_left_complete', 'side_right_complete']

app = FastAPI()

# Load the model
model = tf.keras.models.load_model(MODEL_FILE_PATH)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    # Save the file
    #with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        #f.write(contents)

    # Process the image and make predictions
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)

    # Assuming the model is a classifier with softmax activation, get the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = LABELS[predicted_class_index]

    return {"predicted_class": predicted_class}
