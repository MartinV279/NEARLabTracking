import cv2 as cv
import numpy as np
import operator
from PIL import Image, ImageDraw
import colorsys
import Reinit
# import matplotlib.pyplot as plt
import time

# common elements
lk_params = dict(winSize=(31, 31), maxLevel=5)
sift = cv.KAZE_create(threshold=0.0001, nOctaves=4,
                      diffusivity=cv.KAZE_DIFF_PM_G2)  # cv.xfeatures2d.SURF_create( hessianThreshold=400) #


def init(old_gray1, pts1):  # , gray_orig, old_gray, a):

    (p2, l) = sift.detectAndCompute(old_gray1, None)
    p01 = np.array([[[1, 1]]])

    # img_gray = a.copy()

    for corner in p2:
        p01 = np.append(p01, [[[np.int0(corner.pt[0]), np.int0(corner.pt[1])]]], axis=0)

    p0 = np.delete(p01, 0, 0)
    out = p0.astype(np.float32)
    return out, pts1.reshape(-1, 1, 2).astype(np.float32)
    # out = sift features found
    # pts1 = input argument points reshaped  and parsed for numpy


def do(old_gray, frame_gray, frame_proc, temp, Buffer, Buffer_images, orig1, totalF, Score, p0, pts, jac, trackers,
       threshR=3, flagE=True, t_l=0.0313):
    # all except last 2 are described in main below function call
    # flagE = flag to define if edges should be detected
    # t_l = coefficient for Canny edge detection

    # pts1 = None

    t_h = 2.5 * t_l
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # pyramidal optical flor calculation using Lucas Kanade algorithm. Feature tracking

    img = orig1.copy()  # img = copy of original image

    # Select good points
    good_new = p1[st == 1]
    good_new = np.round(good_new)
    good_old = p0[st == 1]

    if pts is not None and p1.shape[0] > 0.85 * totalF:

        M, mask = cv.findHomography(good_old, good_new, cv.RANSAC, ransacReprojThreshold=threshR)  # ,5.0)

        pts = cv.perspectiveTransform(pts, M)
        print(pts)

        # needs to be restructured from here
        maxx = np.int0(round(np.amax(pts[:, 0, 0])))
        minx = np.int0(round(np.amin(pts[:, 0, 0])))
        maxy = np.int0(round(np.amax(pts[:, 0, 1])))
        miny = np.int0(round(np.amin(pts[:, 0, 1])))
        if minx < 0:
            minx = 0
        if miny < 0:
            miny = 0
        if maxy > frame_gray.shape[0]:
            maxy = frame_gray.shape[0]
        if maxx > frame_gray.shape[1]:
            maxx = frame_gray.shape[1]

        # if minx > 0 and miny>0 and maxy < frame_gray.shape[0] and maxx < frame_gray.shape[1]:
        ar1 = np.asarray(pts, dtype='float32')
        img1 = Image.new('L', (old_gray.shape[1], old_gray.shape[0]), 0)
        ImageDraw.Draw(img1).polygon(ar1, outline=1, fill=1)
        template = (temp * img1)[miny:maxy, minx:maxx]
        # to here

        if len(good_new) > totalF * 0.6:
            Buffer_images[10] = Buffer_images[11]
            Buffer_images[11] = Buffer_images[12]
            Buffer_images[12] = template
            start_time = time.time()

            cv.imwrite('b10.jpg', Buffer_images[10])
            cv.imwrite('b11.jpg', Buffer_images[11])
            cv.imwrite('b12.jpg', Buffer_images[12])
            print("--- %s seconds ---" % (time.time() - start_time))
            sift = cv.KAZE_create(threshold=0.0001, nOctaves=4,
                                  diffusivity=cv.KAZE_DIFF_PM_G2)  # cv.xfeatures2d.SURF_create( hessianThreshold=400) #

            (kp3, des3) = sift.detectAndCompute(template, None)

            pp = sorted(kp3, key=operator.attrgetter('response'), reverse=True)
            sum3 = sum(node.response for node in pp[:20])
            clusterSize = 0

            if len(kp3) > 20:
                if len(kp3) > 0.9 * Buffer["1"][0]:
                    n = 1
                    name = 'b' + str(n) + '.jpg'
                    cv.imwrite(name, template)
                    Buffer_images[n] = template
                    Buffer[str(n)] = [len(kp3), sum3, clusterSize]

                elif len(kp3) > 0.8 * Buffer["2"][0]:
                    n = 2
                    name = 'b' + str(n) + '.jpg'
                    cv.imwrite(name, template)
                    Buffer_images[n] = template
                    Buffer[str(n)] = [len(kp3), sum3, clusterSize]

                elif len(kp3) > 0.7 * Buffer["3"][0]:
                    n = 3
                    name = 'b' + str(n) + '.jpg'
                    cv.imwrite(name, template)
                    Buffer_images[n] = template
                    Buffer[str(n)] = [len(kp3), sum3, clusterSize]

                elif sum3 > 0.9 * Buffer["4"][1]:
                    print(sum(node.response for node in kp3))
                    n = 4
                    name = 'b' + str(n) + '.jpg'
                    cv.imwrite(name, template)
                    Buffer_images[n] = template
                    Buffer[str(n)] = [len(kp3), sum3, clusterSize]

                elif sum3 > 0.8 * Buffer["5"][1]:
                    n = 5
                    name = 'b' + str(n) + '.jpg'
                    cv.imwrite(name, template)
                    Buffer_images[n] = template
                    Buffer[str(n)] = [len(kp3), sum3, clusterSize]
                elif sum3 > 0.7 * Buffer["6"][1]:
                    n = 6
                    name = 'b' + str(n) + '.jpg'
                    cv.imwrite(name, template)
                    Buffer_images[n] = template
                    Buffer[str(n)] = [len(kp3), sum3, clusterSize]

        mask = np.array(mask, dtype=bool)
        mask = np.invert(mask)

        if len(pts) > 0:
            # this part writes debug data on the image and draws the polygon
            pts1 = pts.reshape((-1, 1, 2))
            pts1 = pts1.astype(int)

            cv.polylines(img, [pts1], True, (0, 0, 255), thickness=2)

            newX = np.ma.array(good_new, mask=np.column_stack((mask, mask)))
            newX = newX[~newX.mask]

            p0 = newX.reshape(-1, 1, 2)

            cv.putText(img, 'TRACKING', (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.putText(img, 'Jaccard: ' + str(round(Score, 2)), (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            jac.append(round(Score, 2))
            if Score is not None:
                if Score > 0.75:
                    cv.circle(img, (10, 60), 8, (0, 255, 0), -1)
                elif Score > 0.6:
                    cv.circle(img, (10, 60), 8, (0, 255, 255), -1)
                else:
                    cv.circle(img, (10, 60), 8, (0, 0, 255), -1)
            else:
                cv.circle(img, (10, 60), 8, (0, 0, 255), -1)
    else:
        cv.putText(img, 'RE-INITIALIZING', (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.circle(img, (10, 60), 8, (0, 0, 255), -1)

    h = (p0.shape[0] / totalF) * 0.34
    trackers.append(p0.shape[0])
    t = colorsys.hsv_to_rgb(h, 1, 1)
    cv.circle(img, (10, 80), 8, (255 * t[2], 255 * t[1], 255 * t[0]), -1)

    if len(good_new) < 30 or p1.shape[0] <= 0.85 * totalF or pts is None:  # and flagg == False:
        # f = open(namevid + "tracktime.txt", "a+")
        # f.write("%d\r\n" % tracktime)
        # f.close()
        # tracktime = 0
        # flagg = True
        print('REINIIIITTTT')
        try:

        # pts = vertices of reinitialized contour/polygon/active constraint
        # Score = Jaccard score of reinit
        # Buffer = defined previously
        # Buffer_images = defined previously
            pts, Score, Buffer, Buffer_images = Reinit.create(frame_proc, Buffer, Buffer_images)
        # frame_proc
        # Buffer = defined previously
        # Buffer_images = defined previously

        except:
            print("Reintialization error")

        # pts = None
        # Score = None
        print('Izleze od REINIT')
        if pts is not None:
            # print(pts)

            old_frame = orig1.copy()
            img = orig1.copy()

            # ar = pts
            # pts1 = np.asarray(ar)
            # tim = []

            old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

            if flagE:
                edges = cv.Canny(old_gray, t_l, t_h)
                old_gray = old_gray * edges
            ar1 = np.asarray(pts, dtype='float32')

            img1 = Image.new('L', (old_gray.shape[1], old_gray.shape[0]), 0)
            ImageDraw.Draw(img1).polygon(ar1, outline=1, fill=1)
            mask = np.array(img1)
            print(ar1)
            old_gray1 = mask * old_gray

            sift = cv.KAZE_create(threshold=0.0001, nOctaves=4,
                                  diffusivity=cv.KAZE_DIFF_PM_G2)  # cv.xfeatures2d.SURF_create( hessianThreshold=400) #
            (p2, l) = sift.detectAndCompute(old_gray1, None)
            if len(p2) > 8:
                p01 = np.array([[[1, 1]]])

                # frame1 = old_frame.copy()
                for corner in p2:
                    # x = np.int0(corner.pt[0])
                    # y = np.int0(corner.pt[1])
                    p01 = np.append(p01, [[[np.int0(corner.pt[0]), np.int0(corner.pt[1])]]], axis=0)

                totalF = p01.shape[0]

                p0 = np.delete(p01, 0, 0)
                p0 = p0.astype(np.float32)

                # pts1 = pts.reshape((-1, 1, 2))
                # pts1 = pts1.astype(int)
    return img, p0, pts, Buffer, Buffer_images, totalF, Score, jac, trackers
