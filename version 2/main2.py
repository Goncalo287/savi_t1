# import libraries
import cv2
import copy
from Track import Detection, Track, computeIOU
from random import randint
import numpy as np

import tkinter as tk
from tkinter import simpledialog



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
        haar_face_detections = detector.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=5,
                                            minSize=(70, 70), flags=cv2.CASCADE_SCALE_IMAGE)

        # -----
        # Create list of detections
        # -----   

        
        detection_idx = 0
        for x,y,w,h in haar_face_detections:
            detection_id=str(frame_number) + '-' + str(detection_idx)
            detection = Detection(x, x+w ,y ,y+h , detection_id,frame_stamp,img_gray)
            detections.append(detection)
            detection_idx += 1

        all_detections = copy.deepcopy(detections)

        
        idxs_detections_to_remove = []
        for idx_detection, detection in enumerate(detections):
            for track in tracks:
                if not track.active:
                    continue
                # ----------------
                # Using distance between centers
                # ----------------
                # How to measure hoe close a detection is to a tracker?
                # distance = math.sqrt((detection.cx-track.detections[-1].cx)**2+
                #                      (detection.cx-track.detections[-1].cx)**2)

                # if distance < distance_threshold: # this detection is this tracker!!
                #     track.update(detection) # add detection to tracker
                #     idxs_detections_to_remove.append(idx_detection)
                #     break # do not test this detection with any other track

                # ----------------
                # Using IOU
                # ----------------
                iou = computeIOU(detection, track.detections[-1])
                #print('IOU( ' + detection.detection_id + ' , ' + track.track_id + ') = ' + str(iou))
                if iou > iou_threshold: # This detection belongs to this tracker!!!
                    track.update(detection) # add detection to track
                    idxs_detections_to_remove.append(idx_detection)
                    break # do not test this detection with any other track


        idxs_detections_to_remove.reverse()
        for idx in idxs_detections_to_remove:
            del detections[idx]


        # -----
        # Create new trackers
        # -----

        for detection in detections:
            
            color = [randint(0,255),randint(0,255),randint(0,255)]         

            track = Track(str(person_count),detection,color=color)
            tracks.append(track)
            person_count += 1



        # --------------------------------------
        # Desactivate trackers if last detection has been seen a long time ago
        # --------------------------------------
        for track  in tracks:
            time_since_last_detection = frame_stamp - track.detections[-1].time_stamp
            if time_since_last_detection > desactivate_threshold:
                track.active = False
                track.color = [75,75,75]

        # ---
        # See if track as a detection associated - if not use track()
        # ---

        # draw list of detections
        for detection in all_detections:
            detection.draw(img_gui,(0,0,255))

        # draw list of tracks
        for track in tracks:
            if not track.active:
                continue
            track.draw(img_gui)


    # -----
    # Vizualisation
    # -----
        if frame_number == 0:
                cv2.namedWindow('GUI',cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('GUI',width*2,height*2)

        # show video
        cv2.imshow('GUI',img_gui)
        if cv2.waitKey(25) & 0xFF == ord('q') :
            break

        frame_number += 1

if __name__ == "__main__":
    main()