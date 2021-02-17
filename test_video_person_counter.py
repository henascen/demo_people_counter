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
import time
import cv2

from norfair import Detection, Tracker, draw_tracked_objects

from modules.trackableobject import TrackableObject

from timeit import default_timer as timer

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
# %%
# Tracking distance function
def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

# Argument definitions
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=3, help="# of skip frames between detections")
args = vars(ap.parse_args())

# Defining input
if not args.get("input", False):
    print("[INFO] Comenzando video stream")
    vs = VideoStream('rtsp://admin1:12345ABcd@192.168.1.64/H264?ch=1&subtype=0').start()
    time.sleep(2.0)
else:
    print("[INFO] Abriendo video...")
    vs = cv2.VideoCapture(args["input"])

writer = None

W = None
H = None

totalFrames = 0
people = 0

centroids_nor = []
max_distance_between_points = 100
trackableObjects = {}

total_in = 0

# Dimensions and positions for the in and out boxes that
# condition if the person has entered or exited
out_x1, out_x2, out_y1, out_y2 = 150, 400, 100, 690
in_x1, in_x2, in_y1, in_y2 = 550, 750, 30, 690

fps = FPS().start()

# Tracking iniatialization
tracker = Tracker(distance_function=euclidean_distance, distance_threshold=max_distance_between_points,
                  hit_inertia_min=5, hit_inertia_max=50, point_transience=10)


def get_centroid(tf_box, img_height, img_width):
    x1 = tf_box[1] * img_width
    y1 = tf_box[0] * img_height
    x2 = tf_box[3] * img_width
    y2 = tf_box[2] * img_height
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


while True:
    start = timer()
    centroids_nor = []

    image_np = vs.read()
    image_np = image_np[1] if args.get("input", False) else image_np

    if args["input"] is not None and image_np is None:
        break

    if W is None and H is None:
        (H, W) = image_np.shape[:2]

    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # drawing rec_out
    cv2.rectangle(image_np, (out_x1, out_y1), (out_x2, out_y2), (10, 120, 200), 2)
    # drawing rec_in
    cv2.rectangle(image_np, (in_x1, in_y1), (in_x2, in_y2), (200, 120, 200), 2)

    if totalFrames % args["skip_frames"] == 0:
        image_np_expanded = np.expand_dims(image_np, axis=0)
        input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)
        label_id_offset = 1

        # %%
        # Identifying only a person
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy()
        classes_int = (classes + label_id_offset).astype(int)
        scores = detections['detection_scores'][0].numpy()

        boxes_valid = boxes[scores > 0.7]
        classes_int_valid = classes_int[scores > 0.7]
        scores_valid = scores[scores > 0.7]

        for box in boxes_valid:
            centroids_nor.append(get_centroid(box, H, W))

        detections_nor = [Detection(point) for point in centroids_nor]
        tracked_objects = tracker.update(detections=detections_nor, period=args["skip_frames"])

    else:
        tracked_objects = tracker.update()

    draw_tracked_objects(image_np, tracked_objects, radius=10, id_size=2)

    for person in tracked_objects:
        # print(person.id)
        # print(person.estimate[0])

        to = trackableObjects.get(person.id, None)

        if to is None:
            to = TrackableObject(person.id, person.estimate[0])
        else:
            x = [c[0] for c in to.centroids]
            y = [c[1] for c in to.centroids]
            direction_x = person.estimate[0][0] - np.mean(x)
            direction_y = person.estimate[0][1] - np.mean(y)
            direction_pre_x = person.estimate[0][0] - to.centroids[-1][0]
            direction_pre_y = person.estimate[0][1] - to.centroids[-1][1]

            to.centroids.append(person.estimate[0])
            to.counted = True

            if direction_x > 10 and in_x1 < person.estimate[0][0] < in_x2 and in_y1 < person.estimate[0][1] < in_y2 \
                    and not to.counted_in and to.counted_pre_in:
                total_in += 1
                to.counted_in = True
                to.counted_out = False
                to.counted_pre_in = False
            # direction_x > 10 (direction with inverting)
            if direction_x < -10 and out_x1 < person.estimate[0][0] < out_x2 and out_y1 < person.estimate[0][1] < out_y2 \
                    and total_in > 0 and not to.counted_out and to.counted_pre_out:
                total_in += -1
                to.counted_out = True
                to.counted_in = False
                to.counted_pre_out = False
            # direction_pre_x < -10 (with inverting)
            if direction_pre_x > 5 and out_x1 < person.estimate[0][0] < out_x2 and out_y1 < person.estimate[0][1] < out_y2 \
                    and not to.counted_pre_in:
                to.counted_pre_in = True
            if direction_pre_x < -5 and in_x1 < person.estimate[0][0] < in_x2 and in_y1 < person.estimate[0][1] < in_y2 \
                    and not to.counted_pre_out:
                to.counted_pre_out = True

        trackableObjects[person.id] = to

    # constructing the info tuple with relevant information
    info = [
        ("People inside ", total_in),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(image_np, text, (900, ((i * 40) + 120)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if writer is not None:
        writer.write(image_np)

    # Display output
    cv2.imshow('People counter', cv2.resize(image_np, (600, 600)))

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    totalFrames += 1
    fps.update()

    end = timer()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if not args.get("input", False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()


