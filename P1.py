import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque

from helper import *

def process_videos():
    #Function that performs the video lane Detection 
    #video processing (ref: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html)
    output_dir="video_output/"
    input_dir="test_video"
    create_directory("video_output")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for video_name in os.listdir(input_dir):
        cap = cv2.VideoCapture(input_dir+"/"+video_name)
        out = cv2.VideoWriter(output_dir+video_name,fourcc, 20.0, (960,540)) 
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                image_line = process_image_pipeline(frame)
                out.write(image_line)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
def process_images():
    #Function that performs the image lane Detection 
    output_dir="image_output/"
    input_dir="test_images"
    create_directory("image_output")

    for image_name in os.listdir(input_dir):
        image = mpimg.imread(input_dir+"/"+image_name)
        image_line = process_image(image) 
        cv2.imwrite(output_dir+image_name, cv2.cvtColor(image_line, cv2.COLOR_RGB2BGR))
        
def process_images_pipeline():
    #Function that performs the image pipeline lane Detection
    output_dir="image_output_pipeline/"
    input_dir="test_images"
    create_directory("image_output_pipeline")

    for image_name in os.listdir(input_dir):
        image = mpimg.imread(input_dir+"/"+image_name)
        image_line = process_image_pipeline(image) 
        cv2.imwrite(output_dir+image_name, cv2.cvtColor(image_line, cv2.COLOR_RGB2BGR))

#Image + Video lane detection
process_images() # lane detection
process_images_pipeline() # image lane detection pipeline
process_videos() # video lane detection pipeline
    
