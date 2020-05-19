#!/usr/bin/env python

# Created by Martin Vasilkovski on 21.07.2019
##########################################################################

from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String, Int16MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import Drawing
import Preprocessing
import Tracking
import BufferStartUp
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


#THIS IS ABOUT THE LAYOUT
'''


msg = Int16MultiArray()
a = np.array([[[537, 264]], [[559, 221]], [[533, 157]], [[495, 148]], [[504, 225]], [[519 ,272]]])

a1 = np.reshape(a, len(a)*len(a[0][0]))#, order='F')

msg.data = a1

# This is almost always zero there is no empty padding at the start of your data
msg.layout.data_offset = 0

# create two dimensions in the dim array
msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]

# dim[0] is the vertical dimension of your matrix
msg.layout.dim[0].label = "axes"
msg.layout.dim[0].size = len(a[0][0])
msg.layout.dim[0].stride = len(a)*len(a[0][0])
# dim[1] is the horizontal dimension of your matrix
msg.layout.dim[1].label = "vertices"
msg.layout.dim[1].size = len(a)
msg.layout.dim[1].stride = len(a)
print(msg)
'''

initial_flag = False
mydatetime = datetime.now()
now = mydatetime.strftime('%m%d%H%M')

Buffer_images = []
Buffer = {}
p0 = []
ratio = 0.4
###################################     1   #################################################################
#############   Definition of location of video. Initialize video object to read this video

contour = []
contour1 = []

vid1 =  '' #/home/martin/Desktop/'
namevid = 'a1' #'videoLeft1'
format = '.mp4'

#vid = vid1 + namevid + format
#cap = cv.VideoCapture(vid)
#############################################################################################################
############    To read one fram from the video
#############################################################################################################
#ret, old_frame1 = cap.read()
#############################################################################################################




class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("tracking",Image)
        self.contour_pub = rospy.Publisher("/AC_Path_tracked",Int16MultiArray)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/stereo/left/image_rect_color",Image,self.callback)
        self.contour_sub = rospy.Subscriber("/AC_Path",Int16MultiArray,self.callback1)

    def callback1(self, msg):
        global contour
        global contour1
        contour_t = msg.data
        #print(contour1)
        #print(type(contour1))
        #print(len(contour1))
        contour = np.zeros((len(contour_t)/2,2))
        contour1 = []
        for i in range(len(contour_t)/2): 
            contour[i][0] = contour_t[i*2]*ratio
            contour[i][1] = contour_t[i*2+1]*ratio
            contour1.append((contour_t[i*2]*ratio, contour_t[i*2+1]*ratio))
        contour = np.array(contour)
        #contour1 = np.asarray(contour1, dtype=np.float32)
        #print(contour)
        #print(type(contour))
        #print(len(contour))
            

    def callback(self,data):
        global initial_flag
        global contour
        #global contour1
        global Buffer
        global Buffer_images
        global old_gray
        global totalF
        global Score
        global p0
        global pts
        global jac
        global trackers
        global threshR
        global pts1
        global temp
        #cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        old_frame1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        #plt.imshow(old_frame1)
        #plt.show()
        if not initial_flag:
            
######################################################### REFERENCE FOR TABBING
            
            old_frame = cv.resize(old_frame1, None, fx=ratio, fy=ratio)
            #plt.imshow(old_frame)
            #plt.show()
            #old_frame_c = old_frame.copy()
            means = []
            pts1 = contour #Drawing.Draw(old_frame)
            print("PTS 1 : ")
            #print(pts1)
            print("PTS CONTOUR1 : ")
            #print(contour1)
            #print("=========================================")
            #print(pts1)
            #print("=========================================")

            old_gray, old_gray1, temp, gray_orig = Preprocessing.Track_init(old_frame, contour1)
            #   old_gray = frame rescaled, processed, with edge mask
            #   old_gray1 = frame rescaled, processed, with edge mask and with mask over ROI
            #   temp = frame rescaled, processed, cropped over boundingbox of ROI. Template for model
            #   gray_orig = frame rescaled, processed

            #plt.imshow(old_gray1)
            #plt.show()
            Buffer, Buffer_images = BufferStartUp.start(Buffer, Buffer_images, temp)
            # Buffer = buffer data
            # Buffer_images = list of model images
            
            p0, pts = Tracking.init(old_gray, pts1) # p0 are detected features, pts are vertices of the polygons
            # p0 = kaze features found
            # pts = input argument of vertices/points of the contour is reshaped and parsed for numpy

            #print(p0)
            #print(pts)  
      
            #############   Initialize various variables to be used
            #############################################################################################################


            totalF = len(p0)  # total number of features found/detected at (re-)initialization

            Score = 1  # jaccard score of most recent (re-)initialization
            threshR = 10  # tracker ransacReprojThreshold
            jac = []  # jaccard score history
            trackers = []  # list of active trackers
            img_array = []  # not used
            times = []  # execution times
            initial_flag = True
        else:
            ##########################################   2   ############################################################
            #############   Just a small test to see if the input is an image, and if it is non NULL
    	    

            try:
                frame1 = cv.resize(old_frame1, None, fx=ratio, fy=ratio)
                # resize by predefined ratio
                #frame_orig = frame1.copy()
                # copy
            except:
                print("")

            start = time.time()
            frame_gray, frame_o, frame_proc, temp = Preprocessing.Track(frame1)
            #   frame_gray = frame rescaled, processed, with edge mask
            #   frame_o = original input frame is being sent back
            #   frame_proc = frame rescaled, processed
            #   temp = frame rescaled, processed, cropped over boundingbox of ROI. Template for model. Returned regardless of edge.

            ######### This function called "Tracking.do" does the tracking and reinitialization. There is a LOT of code inside.
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
                Tracking.do(old_gray, frame_gray, frame_proc, temp, Buffer, Buffer_images, frame_o, totalF, Score, p0, pts,
                            jac, trackers, threshR)
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

    ##########################################   3   ############################################################



    #############################################################################################################

    #############################################################################################################
    ###########     Just computes current fps, and prints it on the image.
            t = 1/(time.time() - start)
            cv.putText(img, 'FPS: ' + str(t), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            old_gray = frame_gray.copy()



        #except CvBridgeError as e:
        #    print(e)



        try:
            #self.contour_pub.publish(pts)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
            print(e)

def main(args):
  
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

