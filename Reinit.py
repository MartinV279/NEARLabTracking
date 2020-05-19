import numpy as np
import cv2
import sys
from sklearn import cluster
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese)
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
from sklearn.metrics import jaccard_similarity_score  # jaccard_score
import operator
import time

from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

orb = cv2.KAZE_create(threshold=0.0001, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2)


# startime = time.time()
# print(time.time() - startime)


def store_evolution_in(lst):
    def _store(x):
        lst.append(np.copy(x))

    return _store


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def contrast_enhance(img1):
    lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def create(imgg, Buffer, Buffer_images):
    img2 = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.equalizeHist(img2)
    kp2, des2 = orb.detectAndCompute(img2, None)

    points = np.array([[[1, 1]]])
    for i in range(13):
        img1 = Buffer_images[i]
        ii = i

        kp1, des1 = orb.detectAndCompute(img1, None)
        matches1 = bf.match(des1, des2)
        matches1 = sorted(matches1, key=lambda x: x.distance)

        matches = matches1[:50]

        good_new = np.array([[[1, 1]]])
        good_old = np.array([[[1, 1]]])

        for i in range(0, len(matches)):
            # print('VLEZE VO TOCKI')
            x = np.float32(kp1[matches[i].queryIdx].pt[0])
            y = np.float32(kp1[matches[i].queryIdx].pt[1])
            good_new = np.append(good_new, [[[x, y]]], axis=0)

            x = np.float32(kp2[matches[i].trainIdx].pt[0])
            y = np.float32(kp2[matches[i].trainIdx].pt[1])
            good_old = np.append(good_old, [[[x, y]]], axis=0)

        good_new = np.delete(good_new, 0, 0)
        good_old = np.delete(good_old, 0, 0)
        try:
            M, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, ransacReprojThreshold=25)  # 15

            mask = np.array(mask, dtype=bool)
            mask = np.invert(mask)

            newX = np.ma.array(good_old, mask=np.column_stack((mask, mask)))

            newX = newX[~newX.mask]
            newX1 = newX.reshape(-1, 1, 2)
        except:
            print(ii)

        points = np.append(points, newX1, axis=0)
    points, index = np.unique(points, axis=0, return_index=True)

    new1 = points.reshape(-1, 2)

    dbscan = cluster.DBSCAN(eps=50, min_samples=20)

    X = new1

    dbscan.fit(X)

    if hasattr(dbscan, 'labels_'):
        y_pred = dbscan.labels_.astype(np.int)
    else:
        y_pred = dbscan.predict(X)

    # l1 = X[y_pred == 0, 0]
    l1 = np.asarray(X[y_pred == 0, 0], dtype=np.uint32)
    # l2 = X[y_pred == 0, 1]
    l2 = np.asarray(X[y_pred == 0, 1], dtype=np.uint32)

    ll1 = len(l1)
    ll2 = len(l2)
    if ll1 > 5 and ll1 > 5:

        points1 = np.asarray([l1, l2]).transpose()
        hull = ConvexHull(points1, incremental=True)

        polygon = []
        for x, y in points1[hull.vertices]:
            polygon.append((x, y))

        img = Image.new('L', (imgg.shape[1], imgg.shape[0]), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        maskSNAKE = np.array(img)

        frame = imgg.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # YCR_CB)

        a = hsv[:, :, 0]
        image = img_as_float(a)

        init_ls = maskSNAKE
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        init_ls = cv2.dilate(init_ls, ker, iterations=1)

        evolution = []
        callback = store_evolution_in(evolution)

        ls1 = morphological_chan_vese(image, 20, init_level_set=init_ls, lambda1=2, lambda2=0.4, smoothing=0,
                                      iter_callback=callback)

        # ls1 = ls
        ls = ls1.astype(np.uint8)
        kernel = np.ones((16, 16), np.uint8)
        kernel1 = np.ones((8, 8), np.uint8)
        erosion = cv2.morphologyEx(ls, cv2.MORPH_OPEN, kernel)
        erosion = cv2.erode(erosion, kernel1, iterations=1)

        # img1 = Image.fromarray(imgg, 'RGB')

        _, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_TC89_L1)  # contours, hierarchy

        su1 = 0
        su2 = 0

        # needs to be restructured from here
        if len(contours) > 0:
            img_true = np.array(ls1).ravel()
            img_pred = np.array(erosion).ravel()
            iou = jaccard_similarity_score(img_true, img_pred)  # jaccard_score(img_true, img_pred)
            print('ENTERED CONTOURS')
            a = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
            # print(a)
            # print(cv2.contourArea(a[0]))
            maxContour = cv2.contourArea(a[0])
            maxContourData = a[0]
            mask = np.zeros_like(erosion)
            if maxContour > 1:
                cv2.fillPoly(mask, [maxContourData], 1)
                su1 = sum(mask[(l2, l1)])
        else:
            iou = None

        if len(contours) > 1:
            maxContour1 = cv2.contourArea(a[1])
            maxContourData1 = a[1]
            mask1 = np.zeros_like(erosion)

            if maxContour1 > 1:
                cv2.fillPoly(mask1, [maxContourData1], 1)
                su2 = sum(mask1[(l2, l1)])

        if su1 >= su2 and su1 > 20:
            c = maxContourData
            c = c.astype(np.float32)
        elif su2 >= su1 and su2 > 20:
            c = maxContourData1
            c = c.astype(np.float32)
        else:
            c = None
        # to here

        # template = img2

        if c is not None:
            maxx = np.int0(np.amax(c[:, 0, 0]))
            minx = np.int0(np.amin(c[:, 0, 0]))
            maxy = np.int0(np.amax(c[:, 0, 1]))
            miny = np.int0(np.amin(c[:, 0, 1]))

            template = img2 * maskSNAKE
            template = template[miny:maxy, minx:maxx]

            clusterSize = ll1

            kp3, des3 = orb.detectAndCompute(template, None)
            # Sum of strongest 20
            pp = sorted(kp3, key=operator.attrgetter('response'), reverse=True)
            sum3 = sum(node.response for node in pp[:20])
            if len(kp3) > 20:
                pp = sorted(kp3, key=operator.attrgetter('response'), reverse=True)
                sum3 = sum(node.response for node in pp[:20])
                if ll1 > 0.9 * Buffer["7"][2]:
                    n = 7
                    name = 'b' + str(n) + '.jpg'
                    cv2.imwrite(name, template)
                    Buffer_images[n] = template
                    Buffer[str(n)] = [len(kp3), sum3, clusterSize]

                elif ll1 > 0.8 * Buffer["8"][2]:
                    n = 8
                    name = 'b' + str(n) + '.jpg'
                    cv2.imwrite(name, template)
                    Buffer_images[n] = template
                    Buffer[str(n)] = [len(kp3), sum3, clusterSize]

                elif ll1 > 0.7 * Buffer["9"][2]:
                    n = 9
                    name = 'b' + str(n) + '.jpg'
                    cv2.imwrite(name, template)
                    Buffer_images[n] = template
                    Buffer[str(n)] = [len(kp3), sum3, clusterSize]

        else:
            clusterSize = 0

        return c, iou, Buffer, Buffer_images
    else:
        c = None
        iou = None
        return c, iou, Buffer, Buffer_images
