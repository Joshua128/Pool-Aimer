import cv2
import time
import numpy as np
from pool_isolate import get_mask, mask_better
def dummy_func(x):
     pass

img_path = "ball_images/ball_7.jpg"
pic = cv2.imread(img_path)
pic = cv2.resize(pic, (700, 800))
def mask_balls(img):
     
    #img = cv2.resize(img, (700, 800))
    #grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.namedWindow("track")

    #cv2.createTrackbar('thresh', 'track', 0, 255, dummy_func)
    #cv2.createTrackbar('maxVal', 'track', 0, 255, dummy_func)
    #cv2.createTrackbar('constant_c', 'track', 1, 9, dummy_func)
    pool_mask = mask_better(img)
    masked_outBGR = pool_mask #cv2.cvtColor(pool_mask, cv2.COLOR_HSV2BGR)
    gray_masked_out = cv2.cvtColor(masked_outBGR, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_masked_out, cv2.HOUGH_GRADIENT, 1.2, 25,
                                    param1=90, param2=34,
                                    minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    
            
            
    return circles
"""
msk = mask_better(pic)
cv2.namedWindow("Hough Circle Detection")
cv2.createTrackbar('param1', 'Hough Circle Detection', 100, 300, dummy_func)
cv2.createTrackbar('param2', 'Hough Circle Detection', 30, 100, dummy_func)
cv2.createTrackbar('minDist', 'Hough Circle Detection', 20, 100, dummy_func)
cv2.createTrackbar('minRadius', 'Hough Circle Detection', 0, 100, dummy_func)
cv2.createTrackbar('maxRadius', 'Hough Circle Detection', 0, 200, dummy_func)

#cv2.threshold(grey_img, ) #need thresh, max val
grey_masked_out = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
while(True):

        maxVal = cv2.getTrackbarPos('maxVal', 'track')
        thresh = cv2.getTrackbarPos('thresh', 'track')
        constant_c = cv2.getTrackbarPos('constant_c', 'track')
        new_thresh = cv2.adaptiveThreshold(grey_img, maxVal, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,constant_c)
        kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size for closing
        closed_img = cv2.morphologyEx(new_thresh, cv2.MORPH_CLOSE, kernel)

        # Optional: Apply dilation to make the circles solid and thicker
        dilated_img = cv2.dilate(closed_img, kernel, iterations=1)
        
        p1 = cv2.getTrackbarPos('param1', 'Hough Circle Detection')
        p2 = cv2.getTrackbarPos('param2', 'Hough Circle Detection')
        minDist = cv2.getTrackbarPos('minDist', 'Hough Circle Detection')
        minRadius = cv2.getTrackbarPos('minRadius', 'Hough Circle Detection')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'Hough Circle Detection')

        # Detect circles
        circles = cv2.HoughCircles(grey_masked_out, cv2.HOUGH_GRADIENT, 1.2, minDist,
                                    param1=p1, param2=p2,
                                    minRadius=minRadius, maxRadius=maxRadius)
        
        output = pic.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
        

       
        
        cv2.imshow("dsdds", output)
        cv2.imshow("circl mask", msk)
        cv2.imshow("gray mask", grey_masked_out)
        
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


circles = cv2.HoughCircles(grey_img, cv2.HOUGH_GRADIENT, 1, minDist = 20, param1 = 100, param2 =35, minRadius = 0, maxRadius = 0)
cv2.namedWindow("Hough Circle Detection")
cv2.createTrackbar('param1', 'Hough Circle Detection', 100, 300, dummy_func)
cv2.createTrackbar('param2', 'Hough Circle Detection', 30, 100, dummy_func)
cv2.createTrackbar('minDist', 'Hough Circle Detection', 20, 100, dummy_func)
cv2.createTrackbar('minRadius', 'Hough Circle Detection', 0, 100, dummy_func)
cv2.createTrackbar('maxRadius', 'Hough Circle Detection', 0, 200, dummy_func)


if circles is not None:
        print("TAKEN")
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3)
while(True):
    #cv2.imshow("dsd", img)
    p1 = cv2.getTrackbarPos('param1', 'Hough Circle Detection')
    p2 = cv2.getTrackbarPos('param2', 'Hough Circle Detection')
    minDist = cv2.getTrackbarPos('minDist', 'Hough Circle Detection')
    minRadius = cv2.getTrackbarPos('minRadius', 'Hough Circle Detection')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'Hough Circle Detection')

    # Detect circles
    circles = cv2.HoughCircles(grey_img, cv2.HOUGH_GRADIENT, 1, minDist,
                                param1=p1, param2=p2,
                                minRadius=minRadius, maxRadius=maxRadius)

    # Draw circles
    output = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow("dsdds", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cv2.destroyAllWindows()
"""