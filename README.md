# Billiards Aimer / Line Predictor

## About the Project
The goal of this project was to allow users to take an image of themselves with their pool stick aiming at the cue ball and visualize the projected path the ball (and subsequent collision target) will follow.

This was achieved using the following steps:

- Detect balls using YOLOv11 pretrained image weights
- Locate the white cue ball based on mean pixel intensity values
- Detect the cue stick using Canny edge detection
- Output the predicted path using geometric algebra

## Areas to Improve
- Detection of the cue ball and cue stick is not fully reliable yet, especially under poor lighting conditions or glare
- Addition of spin (English) modeling to improve realism

