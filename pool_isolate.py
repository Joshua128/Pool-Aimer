
import cv2
import time
import numpy as np

def nothing(x):
     pass



def get_mask(img):
    lh = ls = lv = 0
    uh = 159
    us = uv = 255
    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])
    #img = cv2.resize(img, (700, 800))
    blurred = cv2.GaussianBlur(img, (9, 9), 2)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  
    # Create mask
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    return result
    
img_path = "ball_images/ball_7.jpg"



"""
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
"""
def mask_better(img):
    
    h, w = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = img.shape[:2]
    features = lab.reshape((-1, 3))               # shape = (h*w, 3)

    # 2) (Optional) Emphasize chroma vs lightness:
    #    This makes a/b differences count more than L differences,
    #    so dark reds and bright reds stay together.
    features[:,1] *= 5.0    # boost 'a' channel
    features[:,2] *= 2.0   # boost 'b' channel

    # 3) Run k-means with K=3 so we get:
    #      • one cluster for background (greens, cloth variations, etc.)
    #      • one for bright red felt
    #      • one for darker red shadows
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5)
    _, labels, centers = cv2.kmeans(
        features,
        K=3,
        bestLabels=None,
        criteria=criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS
    )

    # 4) Reshape labels back to 2D
    labels2d = labels.reshape(h, w)

    # 5) Figure out which two clusters are “reddest”
    #    (by looking at the 'a' coordinate of each center)
    a_vals = centers[:,1]
    # sort cluster indices by how red they are
    red_order = np.argsort(a_vals)
    bright_red   = red_order[-1]
    shadow_red   = red_order[-2]

    # 6) Build an exclusion mask of both red clusters,
    #    then invert to keep everything else
    exclude = ((labels2d == bright_red) | (labels2d == shadow_red)).astype('uint8') * 255
    keep    = cv2.bitwise_not(exclude)

    # 7) (Optional) clean small speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    keep    = cv2.morphologyEx(keep, cv2.MORPH_OPEN, kernel, iterations=1)

    # 8) Apply to your image
    resultz  = cv2.bitwise_and(img, img, mask=keep)
  
    return resultz

#blurred = cv2.GaussianBlur(img, (9, 9), 2)
#hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)     #good values to adjust to is lh = 0 ls = 0 lv = 0 uh = 171 us = 255 uv = 255


"""
while True:
    
    _,img = cap.read()
    #img = cv2.resize(img, (700, 800))
    h, w = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = img.shape[:2]
    features = lab.reshape((-1, 3))               # shape = (h*w, 3)

    # 2) (Optional) Emphasize chroma vs lightness:
    #    This makes a/b differences count more than L differences,
    #    so dark reds and bright reds stay together.
    features[:,1] *= 2.0    # boost 'a' channel
    features[:,2] *= 2.0    # boost 'b' channel

    # 3) Run k-means with K=3 so we get:
    #      • one cluster for background (greens, cloth variations, etc.)
    #      • one for bright red felt
    #      • one for darker red shadows
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5)
    _, labels, centers = cv2.kmeans(
        features,
        K=3,
        bestLabels=None,
        criteria=criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS
    )

    # 4) Reshape labels back to 2D
    labels2d = labels.reshape(h, w)

    # 5) Figure out which two clusters are “reddest”
    #    (by looking at the 'a' coordinate of each center)
    a_vals = centers[:,1]
    # sort cluster indices by how red they are
    red_order = np.argsort(a_vals)
    bright_red   = red_order[-1]
    shadow_red   = red_order[-2]

    # 6) Build an exclusion mask of both red clusters,
    #    then invert to keep everything else
    exclude = ((labels2d == bright_red) | (labels2d == shadow_red)).astype('uint8') * 255
    keep    = cv2.bitwise_not(exclude)

    # 7) (Optional) clean small speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    keep    = cv2.morphologyEx(keep, cv2.MORPH_OPEN, kernel, iterations=1)

    # 8) Apply to your image
    resultz  = cv2.bitwise_and(img, img, mask=keep)
    








    # Show original and result
    cv2.imshow("ball_mask", resultz)
    cv2.imshow("Mask", img)
  

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # ESC to break
        break

cv2.destroyAllWindows()
"""