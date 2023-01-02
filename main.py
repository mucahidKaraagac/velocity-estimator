###################################################################
#   Velocity_Estimation.py
###################################################################
#
#   @Description:   Velocity estimation script for air vehicles
#
#   Version 0.0.1:  "Developed Code added version system."
#                   ...
#                   24 NOVEMBER 2022, 14:12 - "Mücahid Karaağaç"
#
#
#   @Author(s): "Mücahid Karaağaç"
#
#   @Mail(s):   "mucahidkaraagac@gmail.com"
#
#   24 NOVEMBER 2022 Thursday, 14:12.
#
###################################################################

import cv2 as cv
import numpy as np
import time
import math
import statistics as stas

# video source initialization
uri="rtsp://192.168.2.119:554"
latency=30
width=1280 
height=720
gst_in = ("rtspsrc location={} buffer-size=1 latency={} ! rtph264depay ! h264parse ! omxh264dec ! "
            "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)RGBA ! "
            "videoconvert ! appsink max-buffers=1 drop=true").format(uri, latency, width, height)
cap =cv.VideoCapture(gst_in,cv.CAP_GSTREAMER)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10000,
                       qualityLevel = 0.45,
                       minDistance = 3,
                       blockSize = 3 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (20,20),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(1000,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()

# focusing window tests
# old_frame = old_frame[int(80):int(640),int(360):int(920)]
# old_frame = old_frame[int(140):int(580),int(420):int(860)]

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# pixel to angel transform 
pixel_to_angel_d = math.sqrt((1280*1280)+(720*720))/74.43
# print(pixel_to_angel_d)

if __name__ == '__main__' :
    x = 1
    temp = 0
    velo_list = []
    vel_str = str(0)
    while True:
        time_init = time.time()
        
        ret,frame = cap.read()

        # focusing window tests
        # frame = frame[int(80):int(640),int(360):int(920)]
        # frame = frame[int(140):int(580),int(420):int(860)]
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracksQ
        delta_time = time.time() - time_init
        height_z = 150
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)

            #computation of the displacement velocity vector
            old_to_new = math.sqrt((abs(a-c)*abs(a-c))+(abs(b-d)*abs(b-d)))
            # deg to rad
            rad = 0.01745
            v = (rad*height_z*(old_to_new/pixel_to_angel_d))/(delta_time+temp)
            velo_list.append(v)
            # v_err = v*0.83
            # velo_list.append(v_err)
        # print(pixel_to_angel_d)
        # print(delta_time)    
        # print(len(velo_list))

        # filtering data chunck for the supress noise
        # velo_list.sort(reverse=True)
        # std = stas.stdev(velo_list)
        # vel = stas.median(velo_list)
        # if x % 5 == 0:
        #     for data in velo_list :
        #         if data > (vel+std) or data < (vel-std):
        #             velo_list.remove(data)
        # print(velo_list[:10])
        # print("son",velo_list[-3:])
        # print("--------------------------")
        # print("hiz",stas.median(velo_list))
        # if x % 10 == 0:
        #     vel_str = str(float("{:.5f}".format(stas.median(velo_list))))
        #     velo_list = []   
 
        img = cv.add(frame,mask)
        cv.putText(img,vel_str,(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3,2)
        cv.imshow('frame',img)
        if cv.waitKey(1) == ord('q'):
            break
            
        # data knowledge bin controler
        if x % 1 == 0:
            old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)  
            mask = np.zeros_like(frame)
        # if not x % 10 == 0 : 
        #     # Now update the previous frame and previous points
        #     old_gray = frame_gray.copy()
        #     p0 = good_new.reshape(-1,1,2)
        temp = time.time()-time_init
        # print("time : ",temp)
        # break
        
        # out.write(img)
        x+=1

    out.release()
    cap.release()
    cv.destroyAllWindows()