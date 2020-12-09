import numpy as np
import cv2
import os
import random
import json
import matplotlib.pyplot as plt

DATA_FOLDER = "data/output/training/"


def show_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def prepare_data():
    print("Preparing data\n")

    print("Loading image paths\n")

    image_paths = []
    image_folder = DATA_FOLDER + "front_rgb/"
    image_paths.extend([image_folder + "{}".format(i)
                        for i in os.listdir(image_folder)])

    print("Converting data\n")

    X_train = []
    y_train_steer = []
    y_train = []

    for image_path in image_paths:
        cv_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cv_image = cv2.resize(cv_image, (160, 120),
                              interpolation=cv2.INTER_LANCZOS4)
        X_train.append(cv_image)

        control_frame_path = image_path.replace(
            ".png", ".txt").replace("front_rgb/", "control/")
        with open(control_frame_path) as f:
            control_frame = json.load(f)
            y_train.append([control_frame["throttle"],
                            control_frame["steering"]])
            if control_frame["steering"] == 0:
                y_train_steer.append([0])
            else:
                y_train_steer.append([1])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train_steer = np.array(y_train_steer)

    print("Saving data\n")

    np.save("data/X_train", X_train)
    np.save("data/y_train", y_train)
    np.save("data/y_train_steer", y_train_steer)

    print("Finished preparing data\n")


def load_data():
    print("Loading data\n")

    X = np.load("data/X_train.npy", allow_pickle=True)
    y = np.load("data/y_train.npy", allow_pickle=True)
    y_s = np.load("data/y_train_steer.npy", allow_pickle=True)

    print("Finished loading data\n")

    return X, y, y_s
