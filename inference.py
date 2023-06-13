import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from src.utils import IoU

# Load the model and weights
with keras.utils.custom_object_scope({'IoU': IoU}):
    seg_model = keras.models.load_model('saved_model/seg_model.h5')

seg_model.load_weights('saved_model/seg_model_weights.best.hdf5')

def gen_pred(img_dir, img, model):
    rgb_path = os.path.join(img_dir, img)
    img = cv2.imread(rgb_path)
    # Preprocess the image
    resized_img = cv2.resize(img, (256, 256))
    norm_img = resized_img / 255.0
    img = tf.expand_dims(norm_img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv2.imread(rgb_path), pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file path.")
    parser.add_argument("--input", type=str, help="Path to the file.")
    args = parser.parse_args()

    img_dir = args.input
    imgs = os.listdir(img_dir)

    for i in range(len(imgs)):
        # Predict
        img, pred = gen_pred(img_dir, imgs[i], seg_model)
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Image")
        fig.add_subplot(1, 2, 2)
        plt.imshow(pred, interpolation=None)
        plt.axis('off')
        plt.title("Prediction")
        plt.savefig(f"predictions/prediction_{i}.jpg")