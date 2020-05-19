# Created by Martin Vasilkovski on 21.07.2019
##########################################################################


import cv2 as cv
import Drawing
import Preprocessing
import Tracking
import BufferStartUp
import time
from datetime import datetime

mydatetime = datetime.now()
now = mydatetime.strftime('%m%d%H%M')

Buffer_images = []
Buffer = {}
ratio = 0.4

vid1 = ''  # /home/martin/Desktop/'
namevid = 'a1'  # 'videoLeft1'
format = '.mp4'

vid = vid1 + namevid + format
cap = cv.VideoCapture(vid)
ret, old_frame1 = cap.read()

old_frame = cv.resize(old_frame1, None, fx=ratio, fy=ratio)
# old_frame_c = old_frame.copy()
means = []
pts1 = Drawing.Draw(old_frame)

old_gray, old_gray1, temp, gray_orig = Preprocessing.Track_init(old_frame, pts1)
#   old_gray = frame rescaled, processed, with edge mask
#   old_gray1 = frame rescaled, processed, with edge mask and with mask over ROI
#   temp = frame rescaled, processed, cropped over boundingbox of ROI. Template for model
#   gray_orig = frame rescaled, processed
Buffer, Buffer_images = BufferStartUp.start(Buffer, Buffer_images, temp)
# Buffer = buffer data
# Buffer_images = list of model images

p0, pts = Tracking.init(old_gray1, pts1)  # , gray_orig, old_gray, old_frame_c)
# p0 = kaze features found
# pts = input argument of vertices/points of the contour is reshaped and parsed for numpy

totalF = len(p0)  # total number of features found/detected at (re-)initialization

Score = 1  # jaccard score of most recent (re-)initialization
threshR = 10  # tracker ransacReprojThreshold
jac = []  # jaccard score history
trackers = []  # list of active trackers
img_array = []  # not used
times = []  # execution times
tracktime = 0

while (1):
    ret, frame1 = cap.read()
    # read frame
    try:
        frame1 = cv.resize(frame1, None, fx=ratio, fy=ratio)
        # resize by predefined ratio
        frame_orig = frame1.copy()
        # copy
    except:
        break

    start = time.time()
    frame_gray, frame_o, frame_proc, temp = Preprocessing.Track(frame1)
    #   frame_gray = frame rescaled, processed, with edge mask
    #   frame_o = original input frame is being sent back
    #   frame_proc = frame rescaled, processed
    #   temp = frame rescaled, processed, cropped over boundingbox of ROI. Template for model. Returned regardless of edge.

    # What Trackers.do returns
    # img = most recent acquired frame with tracked contour drawn on top
    # p0 = kaze features found
    # pts = input argument of vertices/points of the contour is reshaped and parsed for numpy
    # Buffer = buffer data
    # Buffer_images = list of model images
    # TotalF = total number of features found/detected at (re-)initialization
    # Score = jaccard score of most recent (re-)initialization
    # jac = jaccard score history
    # trackers = trackers that are active and have been tracked from last frame to current frame
    img, p0, pts, Buffer, Buffer_images, totalF, Score, jac, trackers = \
        Tracking.do(old_gray, frame_gray, frame_proc, temp, Buffer, Buffer_images, frame_o, totalF, Score, p0, pts, jac,
                    trackers, threshR)
    # Arguments used for Tracking.do
    # old_gray = image/frame from previous loop cycle. It is frame_gray from old loop
    # frame_gray = current mose recent frame rescaled, processed, with edge mask
    # frame_proc = current most recent frame rescaled, processed
    # temp = current most recent frame rescaled, processed, cropped over boundingbox of ROI. Template for model. Returned regardless of edge.
    # Buffer = buffer data
    # Buffer_images = list of model images
    # frame_o = current most recent original input frame is being sent back
    # totalF = total number of features detected at (re-)intialization. Used as reference to see how much of the initial trackers have been lost/found
    # Score = jaccard score of most recent (re-)initialization
    # p0 = kaze features found
    # pts = input argument of vertices/points of the contour is reshaped and parsed for numpy
    # jac = jaccard score history
    # trackers = trackers that have been tracked to the last frame, from the one before it. Active trackers that we want to try to find in this loop
    # threshR = tracker ransacReprojThreshold

    ###########     "pts"  variable is the x,y coordinates of the tracked contour/AC

    print(pts)
    t = 1 / (time.time() - start)
    cv.putText(img, 'FPS: ' + str(t), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # write data info on image to be shown
    cv.imshow('frame', img)
    # show image

    old_gray = frame_gray.copy()
    # old_gray = image from this loop becomes old for next loop

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
