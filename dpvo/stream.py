import os
import cv2
import yaml
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain

def image_stream(queue, sequence_path, rgb_txt, calibration_yaml):
    """ image generator """

    # Load calibration
    with open(calibration_yaml, 'r') as file:
        lines = file.readlines()
    if lines and lines[0].strip() == '%YAML:1.0':
        lines = lines[1:]

    calibration = yaml.safe_load(''.join(lines))  

    fx, fy, cx, cy = calibration["Camera.fx"],calibration["Camera.fy"],calibration["Camera.cx"],calibration["Camera.cy"]
    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    # Load paths
    image_list = []
    timestamps = []
    with open(rgb_txt, 'r') as file:
        for line in file:
            timestamp, path, *extra = line.strip().split(' ')
            image_list.append(os.path.join(sequence_path, path))
            timestamps.append(float(timestamp))    

    # Load images
    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
       
        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((timestamps[t], image, intrinsics))

    queue.put((-1, image, intrinsics))


def video_stream(queue, imagedir, calib, stride, skip=0):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    assert os.path.exists(imagedir), imagedir
    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()

