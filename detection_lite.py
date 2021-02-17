import os
import numpy as np
import tensorflow as tf
import cv2
import time
from timeit import default_timer as timer

# Building the model file path 
dirname = os.path.dirname(__file__)
MODELS_DIR = os.path.join(dirname, 'models')
MODEL_FOLDER = 'tflite_ssdmobilenet2_fpn_persondetector/saved_model'
MODEL_NAME = 'model_combined.tflite'

PATH_TO_MODEL_FOLDER = os.path.join(MODELS_DIR, MODEL_FOLDER)
PATH_TO_MODEL = os.path.join(PATH_TO_MODEL_FOLDER, MODEL_NAME)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=PATH_TO_MODEL)

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

IMG_DIR = os.path.join(dirname, 'resources/images')
IMG_FILE = 'test_detection_01.jpg'
PATH_TO_IMG = os.path.join(IMG_DIR, IMG_FILE)

# read and resize the image
img = cv2.imread(PATH_TO_IMG)
new_img = cv2.resize(img, (320, 320))
img_expanded = np.expand_dims(new_img, axis=0)

# input_details[0]['index'] = the index which accepts the input
input_tensor = tf.convert_to_tensor(img_expanded, dtype=tf.float32)

# Scaling the tensor / preprocessing the input
preprocessed_image = ((input_tensor/255)*2) - 1
interpreter.set_tensor(input_details[0]['index'], preprocessed_image.numpy())

start = timer()
# run the inference
interpreter.invoke()
end = timer()
print('Elapsed time is ', (end-start)*1000, 'ms')

# output_details[0]['index'] = the index which provides the input
# output_data = interpreter.get_tensor(output_details[0]['index'])

boxes = interpreter.get_tensor(output_details[0]['index'])
classes = interpreter.get_tensor(output_details[1]['index'])
scores = interpreter.get_tensor(output_details[2]['index'])

boxes = boxes[scores > 0.7]

for box in boxes:
    x1 = int(box[1] * 320)
    y1 = int(box[0] * 320)
    x2 = int(box[3] * 320)
    y2 = int(box[2] * 320)
    cv2.rectangle(new_img, (x1, y1), (x2, y2), (100, 120, 210), 2)

cv2.imshow('TFLite', new_img)
cv2.imwrite('results_tflite.jpg', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
