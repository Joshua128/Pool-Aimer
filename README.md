Billiards Aimer/Line predictor

About the project:
The goal of this project was to allow users to take an image of themselves with their poolstick aiming where they intend the hit the white cue ball and in turn show the projected path
the ball it collides with follows. This was done as listed below:

-Detect balls using COCO pretrianed models
-Locate the white ball based off mean pixel values
-Detecting the cue stick using canny edges
-Outputting the predicted path using some algebra

Areas To Improve:
The detection of the cueball/cuestick is not fully reliable yet, especially in situations where lighting is bad and there is glare.
Addition of taking spin into account
