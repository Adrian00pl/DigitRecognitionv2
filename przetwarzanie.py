import keras
from keras.models import load_model
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = load_model("Model1_upgraded.h5")

def predict(InputImg):
    IMG_SIZE = 28
    img_array = cv2.imread(InputImg, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(new_array)
    switcher = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9"
        }
    return switcher.get(prediction.argmax())
        
