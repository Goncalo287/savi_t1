# import libraries
import cv2
import copy
from Track import Tracker, computeIOU
from random import randint
import numpy as np

import tkinter as tk
from tkinter import simpledialog

trackers = []

def main():
    
    # -----
    # Inicitalization
    # -----

    # get video
    cap = cv2.VideoCapture(0)

    # create face detector
    detector_filename = './frontalfacedefault.xml'
    detector = cv2.CascadeClassifier(detector_filename)

    # setup
    frame_number = 0
    person_count = 0
    tracks = []
    detections = []
    desactivate_threshold = 1
    iou_threshold = 0.2
    avgs = []
    idx_face_to_remove = []
    while cap.isOpened():

        # read image
        result, img_rgb = cap.read()
        
        

        if result is False:
            break

        frame_stamp = round(float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000,2)
        height, width, _ = img_rgb.shape
        


        img_rgb = cv2.flip(img_rgb,1) # mirror image
        img_gui = copy.deepcopy(img_rgb) # copy image

        
        
        img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)

        # -----
        # Detect faces 
        # -----
        faces = detector.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=5,
                                            minSize=(70, 70), flags=cv2.CASCADE_SCALE_IMAGE)
        
        
        # template matching of trackers
        # trackers =  list of images
        for track in trackers:
            h,w = track.img_original.shape
            res = cv2.matchTemplate(img_gray,track.img_original,cv2.TM_CCOEFF_NORMED)
            min_val, max_val,min_loc,max_loc = cv2.minMaxLoc(res)

            x,y = max_loc
            print(max_val)

            # it only compares if max_val > 0.8
            if max_val > 0.5:
                for face_idx,face in enumerate(faces):
                # compare resulting image from matching template with a face detection
                    img_compare = (x,x+w,y,y+h) # left, right, top, bottom
                    iou = computeIOU(img_compare, face)
                    #print('IOU( ' + detection.detection_id + ' , ' + track.track_id + ') = ' + str(iou))
                    if iou > iou_threshold: # it means that the detection correspond to a known tracker
                        # correspond detection to tracker
                        # update tracker
                        track.img_last = img_gray[x:x+w,y:y+h]
                        # remove faces
                        idx_face_to_remove.append(face_idx)

        idx_face_to_remove.reverse
        # remove faces to remove 
        for idx in idx_face_to_remove:
            del(faces[idx])
                
        print('faces:' + str(len(faces)))

        # create new tracker
        for face in faces:
            x,y,w,h = face
            cv2.rectangle(img_gui,(x,y),(x+w,y+h),(255,0,0),2)
            img_original = img_gray[x:x+w,y:y+h]
            name = "Person" + str(person_count)
            tracker = Tracker(img_original,img_original,name,x,y,w,h)
            trackers.append(tracker)
            person_count += 1

        print(len(trackers))

        for track in trackers:
            track.draw(img_gui)
    # -----
    # Vizualisation
    # -----
        if frame_number == 0:
                cv2.namedWindow('GUI',cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('GUI',width*2,height*2)

        # show video
        cv2.imshow('GUI',img_gui)
        if cv2.waitKey(0) & 0xFF == ord('q') :
            break

        frame_number += 1

if __name__ == "__main__":
    main()