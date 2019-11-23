from __future__ import division
import cv2
import numpy as np


def nothing(x):
    pass

    """图像二值化"""


icol = (0, 87, 0, 75, 255, 255)
cv2.namedWindow('colorTest')

cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)

frame1 = cv2.imread('YellowStick.jpg')
frame = cv2.medianBlur(frame1, 5)
a = 0
while True:

    lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
    lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
    lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
    highHue = cv2.getTrackbarPos('highHue', 'colorTest')
    highSat = cv2.getTrackbarPos('highSat', 'colorTest')
    highVal = cv2.getTrackbarPos('highVal', 'colorTest')

    frameBGR = cv2.GaussianBlur(frame, (5, 5), 0)

    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)

    colorLow = np.array([lowHue, lowSat, lowVal])
    colorHigh = np.array([highHue, highSat, highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)

    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

    cv2.imshow('mask1', mask1)
    cv2.imwrite('picture.jpg', mask1)
    cv2.imshow('colorTest', frame)

    img = cv2.imread('picture.jpg')
    img0 = cv2.imread('YellowStick.jpg')
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img1, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if a == 0:
            print((x, y, w, h))
            print((x + (1 / 2 * w), y + (1 / 2 * h)))
            a = 1
        cv2.rectangle(img0, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('test2', img0)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()