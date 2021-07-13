import argparse
import io
import os
from typing import List

import cv2
import mss
import numpy as np
import pycountry
import tensorflow as tf
from dotenv import load_dotenv
from google.cloud import vision
from PIL import Image
from train.model import get_model

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_prediction_v1(class_name, prob, *args, **kwargs):
    print(f"{class_name} {prob:.0f}%")


def get_color(prob):
    if prob >= 75:
        return colors.OKGREEN + colors.BOLD
    elif 50 <= prob < 75:
        return colors.WARNING
    else:
        return colors.FAIL


def print_prediction_v2(class_name, prob, previous_class, previous_prob, languages, *args, **kwargs):
    if class_name == "Korea, South" or previous_class == "Korea, South":  # Occurs too often so ignore it.
        return

    languages_text = f"{colors.OKBLUE}{' '.join(languages)}{colors.ENDC}" if len(languages) > 0 else f"{colors.ENDC}"

    if previous_class is not None and class_name != previous_class:
        print(" " * 70, end="\r")
        print(f"{get_color(previous_prob)}{previous_prob:3d}%  {previous_class:25s}  {languages_text}")

    print(f"{get_color(prob)}{prob:3d}%  {class_name:25s}  {languages_text}", end="\r")


def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(gray, cv2.CV_64F).var() < 125.0


def detect_text(image: np.ndarray, client: vision.ImageAnnotatorClient):
    """Detect text in the image."""
    image = Image.fromarray(image)
    output = io.BytesIO()
    image.save(output, format="JPEG")
    image = vision.Image(content=output.getvalue())

    response = client.text_detection(image=image)

    languages = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                if len(paragraph.property.detected_languages) > 0:
                    language_code = paragraph.property.detected_languages[0].language_code
                    try:
                        language = pycountry.languages.get(alpha_2=language_code).name
                        confidence = paragraph.property.detected_languages[0].confidence
                        if confidence > 0.98 and language != "English":
                            languages.append((language, confidence))
                    except Exception:
                        pass

    languages = sorted(languages, key=lambda x: x[1], reverse=True)

    top_languages: List[str] = []
    for language, _ in languages:
        if language not in top_languages:
            top_languages.append(language)
        if len(top_languages) == 3:
            break

    return top_languages


def screen_record(model, dataset):
    box = {"top": 400, "left": 100, "width": 1000, "height": 1000}

    title = "screen"
    sct = mss.mss()

    client = vision.ImageAnnotatorClient()

    cv2.namedWindow(title)
    cv2.moveWindow(title, 420, 1250)

    previous_class, previous_prob = None, None

    while True:
        img = np.asarray(sct.grab(box))
        img = np.array(Image.fromarray(img).resize((640, 640)).convert("RGB"))

        if not is_blurry(img):
            languages = detect_text(img, client)  # TODO: Implement OCR reading on frame.

            prediction = tf.squeeze(model(img[np.newaxis, ...], training=False))

            class_id = tf.argmax(prediction)
            class_name = dataset.class_names[class_id]

            prob = int(round(prediction[class_id].numpy() * 100, 0))
            print_prediction_v2(class_name, prob, previous_class, previous_prob, languages)

            previous_class, previous_prob = class_name, prob

            cv2.imshow(title, np.array(Image.fromarray(img).resize((1000, 1000))))
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Data directory.", required=True)
    parser.add_argument("--model", help="Model checkpoint directory.", required=True)
    args = parser.parse_args()

    model = get_model()
    model.load_weights(f"{args.model}/checkpoint")

    model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    dataset_params = {
        "directory": args.data,
        "image_size": (640, 640),
        "validation_split": 0.2,
        "seed": 123,
    }
    dataset = tf.keras.preprocessing.image_dataset_from_directory(subset="validation", **dataset_params)

    screen_record(model, dataset)


if __name__ == "__main__":
    main()
