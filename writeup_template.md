#**Udacity Project: Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goal of this project is make a pipeline that finds lane lines on the road

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

1) I converted the images to grayscale
2) Then I applied Canny in order to perform the edge detection 
3) 
4)
5) In this step I put in practice the hough transform, in order to draw a single line on the left and right lanes, I modified the draw_lines() function using the following algorithm:
    5.1) In the first step I separated the lines by slope
    
6) Finally the lines "detected" are plot in the image
 
These are some of my results!

![alt text]./image_output_pipeline/solidWhiteRight.jpg[image1]
![alt text]./image_output_pipeline/solidYellowCurve2.jpg[image1]
![alt text]./image_output_pipeline/whiteCarLaneSwitch.jpg[image1]

###2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be the fact that the algorithm is not tolerance a roads with curves, but this is one feature
that I would like to implement in the next version (maybe approaching the challenge?).

Another shortcoming could be how the algorithm is written, due to the lack of experience on python+opencv in the automotive domain (for example variable naming or function naming...). I would like to review and improve the implementation when I will be more confident with opencv+python.

###3. Suggest possible improvements to your pipeline

This is my first version of the lane detection algorithm. I would like to improve the algorithm with the following things:
1) I would like to explore the strategies to perform the lane detection with signals on the road. Maybe modifying the region of interest?
2) Improve the algorithm with less tolerance to roads with slope+curve

