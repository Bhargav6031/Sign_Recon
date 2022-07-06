import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np


def decode(value):
    class_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
                     'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20,
                     'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'nothing': 26}

    key_list = list(class_indices.keys())
    val_list = list(class_indices.values())

    # print key with val 100
    position = val_list.index(value)
    return key_list[position]

# fontScale
def hi(image):
    fontScale = 1

    # Red color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2




    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    model = tf.keras.models.load_model('D:\SE project\machine learning model\sl_model.h5')



    # For webcam input:

    resized = cv2.resize(image, (128,128))
    img_array = np.array([resized])
    prediction = model.predict(img_array)
    pred = decode(np.argmax(prediction))

    cv2.putText(image,pred,(200,200),font, fontScale,
                                color, thickness, cv2.LINE_AA, False)


    # cv2.imshow('MediaPipe Hands', image)
    
    return image
