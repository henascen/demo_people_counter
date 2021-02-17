import os

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# Building the model file path 
dirname = os.path.dirname(__file__)
MODELS_DIR = os.path.join(dirname, 'models')
MODEL_NAME = 'ssdmobilenet2_fpn_persondetector'

PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))

LABEL_FILENAME = 'person_label_map.pbtxt'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))

# %%
# Load the model
# ~~~~~~~~~~~~~~

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


# %%
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# Argument definitions
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=2, help="# of skip frames between detections")
args = vars(ap.parse_args())

# Defining input
if not args.get("input", False):
    print("[INFO] Comenzando video stream")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    print("[INFO] Abriendo video...")
    vs = cv2.VideoCapture(args["input"])

writer = None

W = None
H = None

totalFrames = 0

fps = FPS().start()

while True:

    image_np = vs.read()
    image_np = image_np[1] if args.get("input", False) else image_np

    rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    if W is None and H is None:
        (H, W) = image_np.shape[:2]

    # print(H, W)
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    image_np_with_detections = image_np.copy()

    if totalFrames % args["skip_frames"] == 0:
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1

        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy()
        classes_int = (classes + label_id_offset).astype(int)
        scores = detections['detection_scores'][0].numpy()
        
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes,
            classes_int,
            scores,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.70,
            agnostic_mode=False)

    if writer is not None:
        writer.write(image_np_with_detections)

    # Display output
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (600, 600)))

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    totalFrames += 1

if not args.get("input", False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()