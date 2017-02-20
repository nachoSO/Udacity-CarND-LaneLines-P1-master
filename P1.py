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
                image_line = process_image(frame)
                out.write(image_line)
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
def process_images():
    #image processing
    output_dir="image_output/"
    input_dir="test_images"
    create_directory("image_output")

    for image_name in os.listdir(input_dir):
        image = mpimg.imread(input_dir+"/"+image_name)
        image_line = process_image(image) 
        cv2.imwrite(output_dir+image_name, cv2.cvtColor(image_line, cv2.COLOR_RGB2BGR))

process_images()
process_videos()
    
