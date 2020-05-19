import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
#import matplotlib.pyplot as plt
#from statistics import mean

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)

def contrast_enhance(img1):
    lab = cv.cvtColor(img1, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv.merge((cl, a, b))
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return final

def Reflection(im):
    s = im.shape
    x = int(s[0] / 2)
    y = int(s[1] / 2)
    w = x
    thresh = 0.80
    temp = im[y - w:y + w, x - w:x + w, 0]  # single channel

    # start_time = time.time()

    ret, temp_mask = cv.threshold(temp, thresh * 256, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), 'uint8')
    temp_mask = cv.dilate(temp_mask, kernel)
    im[y - w:y + w, x - w:x + w, :] = cv.inpaint(im[y - w:y + w, x - w:x + w, :], temp_mask, 1, cv.INPAINT_TELEA)

    return im


def cum(hist):
    maxx  = np.amax(hist)
    loo = []
    co = 0
    for a in hist:
        loo.append(a[0] + co)
        co = loo[-1]
    div = loo[-1]/maxx
    loo = loo / div
    return loo




def Track_init(old_frame, ar1, ga = 0.5, flagE = True, t_l = 0.0313, ratio = 1, gaus = 5):
    old_frame = cv.resize(old_frame, None, fx=ratio, fy=ratio)
    t_h = 2.5 * t_l

    old_frame = contrast_enhance(old_frame)

    old_frame = adjust_gamma(old_frame, gamma=ga)

    old_frame = cv.GaussianBlur(old_frame, (gaus, gaus), 0)

    old_gray_t = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(old_gray_t, t_l, t_h)
    if flagE:
        old_gray = old_gray_t * edges

    ar1 = np.asarray(ar1, dtype='float32')
    img = Image.new('L', (old_gray.shape[1], old_gray.shape[0]), 0)
    ImageDraw.Draw(img).polygon(ar1, outline=1, fill=1)
    mask = np.array(img)

    old_gray1 = mask * old_gray

    maxx = np.int0(np.amax(ar1[:, 0]))
    minx = np.int0(np.amin(ar1[:, 0]))
    maxy = np.int0(np.amax(ar1[:, 1]))
    miny = np.int0(np.amin(ar1[:, 1]))
    old = old_gray_t * mask
    template = old[miny:maxy, minx:maxx]

    return old_gray, old_gray1, template, old_gray_t
    #   frame rescaled, processed, with edge mask
    #   frame rescaled, processed, with edge mask and with mask over ROI
    #   frame rescaled, processed, cropped over boundingbox of ROI. Template for model
    #   frame rescaled, processed

def Track(frame, ga = 0.5, flagE = True, flagR = False,  t_l = 0.0313, ratio = 1, gaus = 5):

    frame_orig = frame.copy()
    frame = contrast_enhance(frame)
    if flagR:
        frame = Reflection(frame)

    t_h = 2.5 * t_l
    frame = cv.resize(frame, None, fx=ratio, fy=ratio)

    frame = adjust_gamma(frame, gamma=ga)
    frame = cv.GaussianBlur(frame, (gaus, gaus), 0)
    frame_proc = frame
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame_gray = cv.equalizeHist(frame_gray)

    temp = frame_gray.copy()

    if flagE:
        edges1 = cv.Canny(frame_gray, t_l, t_h)
        frame_gray = frame_gray * edges1
    return frame_gray, frame_orig, frame_proc, temp
    #   frame rescaled, processed, with edge mask
    #   original input frame is being sent back
    #   frame rescaled, processed
    #   frame rescaled, processed, cropped over boundingbox of ROI. Template for model. Returned regardless of edge.
