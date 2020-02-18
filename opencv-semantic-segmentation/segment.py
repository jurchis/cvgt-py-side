# -*- coding: utf-8 -*-
"""
@author: Florin Jurchis
"""

import segment_image
import segment_video
import cv2
import requests
from io import BytesIO
from PIL import Image
import sys
import os

os.chdir(r"C:\Users\Florin Jurchis\Desktop\Training\JAVA\cvgt-py-side\opencv-semantic-segmentation")

def runSegment(file_path):
    while True:
        try:
            response = requests.get(file_path)
            img = Image.open(BytesIO(response.content))
            img.save("img.png","PNG")
            if file_path.split(".")[-1] in ("png", "jpg"):
                image = cv2.imread("img.png")
                print("[INFO] Image loaded successfully.")
                segment_image.segment_image(image, file_path)
                break
            elif file_path.split(".")[-1] in ("mp4", "mkv", "gif"):
                vs = cv2.VideoCapture(file_path)
                print("[INFO] Video loaded successfully.")
                segment_video.segment_video(vs, file_path)
                break
            elif file_path == "":
                break
            else:
                raise Exception("You must select a valid image or video file format.")
                
        except Exception as ex:
            print("[ERROR] No file was loaded")
            print("[ERROR]",ex)
            if ex != "You must select a valid image or video file format.":
                break
            
if __name__ == "__main__":
    file_path=sys.argv[1]
    runSegment(file_path)