#!/usr/bin/env python3

import cv2
import numpy as np
import copy
import tkinter as tk
from tkinter import simpledialog
import time
import math
import json

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


def computeIOU(face_box, tracker_box):

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


def mouseCallback(event,x1,y1,flags,param):
    global trackers
    
    if event == cv2.EVENT_LBUTTONUP:
        for face in unknown_faces:
            x,y,w,h = face
            if x < x1 < (x+w) and y < y1 < (y+h): # mouse in detection
                # create template and define track id
                if w * h > 100:
                    name = openInputWindow()    # Open dialog box to input person's name
                    if name is not None and len(name)>0:
                        new_tracker = Tracker(img_gray[y:y+h, x:x+w], name)
                        trackers.append(new_tracker)
                        print('Template saved:', name)
                    break


def saveTrackers(trackers):
    saved_templates = []
    for idx, tracker in enumerate(trackers):
        img_path = 'templates/template_' + str(idx) + '.png'
        cv2.imwrite(img_path, tracker.img_original)
        saved_templates.append({'name': tracker.name, 'path': img_path})

    with open('templates/list.json', 'w') as file:
        json.dump(saved_templates, file, indent=4)

    print('Saved {} trackers to disk'.format(len(saved_templates)))


def loadTrackers():
    try:
        with open('templates/list.json') as file:
            saved_templates = json.load(file)
    except FileNotFoundError:
        saved_templates = []

    trackers = []
    for tracker in saved_templates:
        img_template = cv2.imread(tracker['path'], cv2.IMREAD_GRAYSCALE)
        if img_template is not None:
            new_tracker = Tracker(img_template, tracker['name'])
            trackers.append(new_tracker)
    
    print('Loaded {}/{} trackers from disk'.format(len(trackers), len(saved_templates)))
    return trackers



# Global variables
trackers = []
# trackers = loadTrackers()
img_gray = None
unknown_faces = []


def main():
    global unknown_faces, img_gray, trackers
    # Initialization
    cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    timeout = 5 # seconds
    match_thresh = 0.6  # 0 -> 1
    iou_thresh = 0.6    # 0 -> 1

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame',mouseCallback)


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
        for tracker in trackers:
            face_detected = False

            # Template matching: find the tracker's saved template in the image
            res = cv2.matchTemplate(img_gray, tracker.img_latest, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, top_left = cv2.minMaxLoc(res)


            # If the best match found isn't good enough, reset
            if max_val < match_thresh:
                tracker.reset()
                continue


            # Save the best match's coordinates in (x, y, w, h) format: same as face detections
            x, y, w, h = top_left[0], top_left[1], tracker.img_latest.shape[1], tracker.img_latest.shape[0]


            # Go through all the detected faces and check if they match this template
            for face_idx, face in enumerate(faces):

                iou = computeIOU(face, (x, y, w, h))    # iou = intersection over union
                if iou > iou_thresh:
                    face_detected = True
                    x, y, w, h = face
                    faces_tracked_idx.append(face_idx)
                    break   # Stop searching after the first hit to avoid finding multiple matching faces


            # If a mathing face was found, (x, y, w, h) are the face's coordinates and, if not, they are the results
            # of the template match. This allows the program to track people even when their face isn't detected.
            tracker.img_latest = img_gray[y:y+h, x:x+w]


            # If a matching face was found, reset the timer. If not, track the template for X seconds (timeout) and then reset.
            if face_detected:
                tracker.last_face_timestamp = time.time()
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0,255,0), 3)
                cv2.putText(img_bgr, str(round(iou*100))+'%', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(img_bgr, tracker.name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

            else:
                # 'time_elapsed' counts down from from 'timeout' to 0
                time_elapsed = timeout - (time.time() - tracker.last_face_timestamp)
                if time_elapsed > 0:
                    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0,255,255), 3)
                    cv2.putText(img_bgr, str(math.ceil(time_elapsed))+'s', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                    cv2.putText(img_bgr, tracker.name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                else:
                    tracker.reset()


        # If a detected face has no associated tracker, highlight is as an 'unknown' face
        unknown_faces = []
        for face_idx, face in enumerate(faces):
            if face_idx not in faces_tracked_idx:
                x, y, w, h = face
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0,0,255), 3)
                cv2.putText(img_bgr, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                unknown_faces.append(face)


        # Visualization
        cv2.imshow('Frame', img_bgr)
        # TODO: 'data base' gallery view (with both original and updated templates)


        # Keyboard inputs
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q') or k == 27:    # Q or ESC to exit
            break

        elif k == ord('t'):     # T to create a new tracker

            x, y, w, h = cv2.selectROI('Frame', img_bgr)
            if w * h > 100:
                name = openInputWindow()    # Open dialog box to input person's name
                if name is not None and len(name)>0:
                    new_tracker = Tracker(img_gray[y:y+h, x:x+w], name)
                    trackers.append(new_tracker)
                    print('Template saved:', name)

        elif k == ord('r'):     # R to refresh (use original image)
            for tracker in trackers:
                tracker.reset()
            print('Trackers refreshed')
        
        elif k == ord('s'):     # S to save trackers
            saveTrackers(trackers)

        elif k == ord('l'):     # L to load trackers
            trackers = loadTrackers()


    # Destroy cv2 windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()