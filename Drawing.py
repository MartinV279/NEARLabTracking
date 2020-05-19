import cv2 as cv
import numpy as np


def Draw(img):
    drawing = False  # true if mouse is pressed
    mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
    ix, iy = -1, -1
    ar = []

    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing, mode

        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)
            ar.append([ix, iy])

    cv.namedWindow('image')
    cv.setMouseCallback('image', draw_circle)

    while (1):
        cv.imshow('image', img)
        k = cv.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
    return np.asarray(ar)