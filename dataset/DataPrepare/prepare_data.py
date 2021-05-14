#! /usr/bin/env python

import cv2
import os
import numpy
import argparse
import time

parser = argparse.ArgumentParser(description="Process video data")
parser.add_argument(
    "-p", "--data_path", required=True, default="", help="Path to video file"
)
parser.add_argument(
    "-o",
    "--output_folder",
    required=False,
    default="/output",
    help="output folder path",
)

args = parser.parse_args()


out_dir = os.path.join(args.data_path.split("/")[0], "Fall_Images_1")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

pathFiles = dict()

for root, dirs, files in os.walk(args.data_path):
    if len(files) > 0:
        pathFiles[root] = files

name_format = "{:05d}.png"

currentFrame = 0
for root, files in pathFiles.items():

    for file in files:

        out_path = os.path.join(out_dir, file.split(".")[0])
        in_path = os.path.join(root, file)
        print("outPath:", out_path)
        print("InPath:", in_path)
        try:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        except OSError:
            print("Error: Creating directory of data")

        cap = cv2.VideoCapture(in_path)
        # cap = cv2.VideoCapture(args.data_path)
        while cap.isOpened():

            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret != True:
                break
            else:
                #print("Frame shape:", frame.shape)
                # Saves images of the current frame in png file
                output_path = os.path.join(
                    out_path, str(name_format.format(currentFrame))
                )
                cv2.imwrite(output_path, frame)
                currentFrame = currentFrame + 1

        # time.sleep(5)
        cap.release()
        currentFrame = 0
