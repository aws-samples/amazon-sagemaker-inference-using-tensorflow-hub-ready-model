import json
import boto3
import json_logging
import os
import logging
import sys
import time
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask
from flask import request


# Settings
if os.environ.get("MIN_SCORE"):
    MIN_SCORE = float(os.environ.get("MIN_SCORE"))
else:
    MIN_SCORE = 0.1

if os.environ.get("MAX_BOXES"):
    MAX_BOXES = int(os.environ.get("MAX_BOXES"))
else:
    MAX_BOXES = 15


# Logger
# Logger initialized
json_logging.init_non_web(enable_json=True)

logger = logging.getLogger("serving")
if os.environ.get("LOG_LEVEL"):
    logger.setLevel(os.environ.get("LOG_LEVEL"))
else:
    logger.setLevel(logging.DEBUG)

logger.addHandler(logging.StreamHandler(sys.stdout))

# Flask app
app = Flask(__name__)

# Load tensor hub model
print("Loading model")
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']
print("Model loaded")


def build_result_object(boxes, class_names, scores, max_boxes=15, min_score=0.1):
    logger.debug("Building {} boxes with min score of {}".format(max_boxes, min_score))
    result_object_list = []
    for i in range(min(boxes.shape[0], max_boxes)):
        logger.debug("checking if {} is bigger than {}".format(scores[i], min_score))
        if scores[i] >= min_score:
            y_min, x_min, y_max, x_max = tuple(boxes[i])
            logger.info("object detected", extra={'props': {'service': 'tf-hub',
                                                            'object': class_names[i].decode("ascii"),
                                                            'confidence': str(scores[i])}})
            obj_dict = {"ymin": str(y_min),
                        "xmin": str(x_min),
                        "ymax": str(y_max),
                        "xmax": str(x_max),
                        "class": class_names[i].decode("ascii"),
                        "confidence": str(scores[i]),
                        "mp3": "_empty_"
                        }
            result_object_list.append(obj_dict)
            logger.debug("Added {} objects to result".format(len(result_object_list)))

    return result_object_list


def download_s3_img(bucket_name, object_prefix, object_key):
    logger.debug("Downloading from {} image {}/{}".format(bucket_name, object_prefix, object_key))
    full_object_path = object_prefix + '/' + object_key
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, full_object_path, object_key)
    # Cant resize, I need to keep the object detection the same as the image received from the client
    # new_width=256, new_height=256, new_quality=100
    #
    # logger.debug("downloaded image from s3")
    # with open(image_file_name, 'rb') as image:
    #     image_data = BytesIO(image.read())
    # logger.debug("loading image")
    # pil_image = Image.open(image_data)
    # logger.debug("resizing image")
    # pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    # logger.debug("converting image to RGB")
    # pil_image_rgb = pil_image.convert("RGB")
    # logger.debug("saving new image")
    # pil_image_rgb.save(image_file_name, format="JPEG", quality=new_quality)
    return object_key


def load_img(path):
    logger.debug("Loading image {}".format(path))
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def run_detector(func_detector, img):
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = func_detector(converted_img)
    end_time = time.time()
    result = {key: value.numpy() for key, value in result.items()}
    logger.info("Found %d objects." % len(result["detection_scores"]))
    elapsed_time = end_time - start_time
    logger.info("Inference time: {}".format(elapsed_time))
    return result


def delete_temp_image(file):
    os.remove(file)


@app.route('/ping')
def ping():
    return "pong"


@app.route('/')
def home():
    return "nothing here"


@app.route('/robots.txt')
def robots():
    return "User-agent: * \n disallow /"


@app.route('/invocations', methods=['GET', 'POST'])
def invocations():
    if request.method == "POST":
        response_body = request.get_json()
        if "file_name" in response_body:
            s3_bucket = response_body['s3_bucket']
            object_prefix = response_body['key_prefix']
            object_key = response_body['file_name']
            downloaded_image_name = download_s3_img(s3_bucket, object_prefix, object_key)
            img = load_img(downloaded_image_name)
            result = run_detector(detector, img)
            result_object = build_result_object(result["detection_boxes"], result["detection_class_entities"],
                                                result["detection_scores"], MAX_BOXES, MIN_SCORE)
            delete_temp_image(object_key)
            return json.dumps(result_object)
        else:
            return "Missing file name in POST request"


if __name__ == '__main__':
    app.run(port=8080, host="0.0.0.0", debug=False)
