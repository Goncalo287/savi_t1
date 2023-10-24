#!/usr/bin/env python3

import cv2
import numpy as np
import copy
import tkinter as tk
from tkinter import simpledialog

from track import Tracker

def openInputWindow():
    '''
    Opens a window where the user can input text
    Returns a string or None (if the user cancels)
    '''

    root = tk.Tk()
    root.withdraw()
    user_input = simpledialog.askstring('Set template name', 'Person name:')
    root.destroy()
    return user_input


def computeIOU(face_box, tracker_box,image):
    


    x, y, w, h = face_box
    x1_1, y1_1, x2_1, y2_1 = x, y, x+w, y+h
    x, y, w, h = tracker_box
    x1_2, y1_2, x2_2, y2_2 = x, y, x+w, y+h


    # Calculate the area of the first bounding box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    # Calculate the area of the second bounding box
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate the coordinates of the intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there is an intersection
    if x1_i < x2_i and y1_i < y2_i:
        # Calculate the area of the intersection
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate the area of the union
        area_u = area1 + area2 - area_i
        
        # Calculate IoU
        iou = area_i / area_u
        return iou
    else:
        return 0.0



def main():

    # Initialization
    cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    templates = []  # TODO: open templates saved on disk (add image format to .gitignore)
    trackers = []



    # Execution
    while cap.isOpened():

        ret, frame = cap.read()
        if ret is False:
            break


        # Image processing
        img_bgr = copy.deepcopy(frame)
        img_bgr = cv2.flip(img_bgr,1) # mirror image
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


        # Detect faces
        faces = face_classifier.detectMultiScale(image=img_gray, scaleFactor=1.2, minNeighbors=4, minSize=(70,70))

        # Update trackers and compare with detected faces
        faces_tracked_idx = []
        for tracker_idx, tracker in enumerate(trackers):
            success, detection_box = tracker.tracker.update(img_bgr)
        

            if success:

                x, y, w, h = [int(v) for v in detection_box]
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img_bgr,'Person ' + str(tracker_idx), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)


                for face_idx, face in enumerate(faces):
                    iou = computeIOU(face, detection_box,img_bgr)
                    if iou > 0.1:
                        faces_tracked_idx.append(face_idx)
            else:
                print('failed')

        # Create a new tracker for every face that isn't tracked
        for face_idx, face in enumerate(faces):
            if face_idx not in faces_tracked_idx:

                '''
                List of trackers:
                cv2.TrackerCSRT_create - works
                cv2.TrackerKCF_create
                cv2.TrackerBoosting_create
                cv2.TrackerMIL_creat,
                cv2.TrackerTLD_create
                cv2.TrackerMedianFlow_create
                cv2.TrackerMOSSE_create
                '''
                tracker = cv2.TrackerCSRT_create()

                initBB = face
                tracker.init(img_bgr, initBB)
                x,y,w,h = face
                new_tracker = Tracker(tracker,initBB,img_bgr[y:y+h,x:x+w]," ")
                trackers.append(new_tracker)

                

        # Visualization
        cv2.imshow('Frame', img_bgr)


        # Keyboard inputs
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q') or k == 27:    # Q or ESC to exit
            break

        elif k == ord('s'):     # S to save current face as a template

            if len(faces) > 0:
                name = openInputWindow()    # Open dialog box to input person's name
                if name is not None and len(name)>0:
                    new_template = {'name': name, 'img': img_gray[y:y+w, x:x+w]}    # (x, y, w, h) from face detection
                    templates.append(new_template)
                    print('Template saved:', name)
            else:
                print('Error: no faces detected')


    # Destroy cv2 windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()