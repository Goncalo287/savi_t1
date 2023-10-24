#!/usr/bin/env python3

import cv2
import numpy as np
import copy
import tkinter as tk
from tkinter import simpledialog

def computeIOU(d1, d2):
    # box1 and box2 should be in the format (x1, y1, x2, y2)
    x1_1, y1_1, x2_1, y2_1 = d1.left, d1.top, d1.right, d1.bottom
    x1_2, y1_2, x2_2, y2_2 = d2.left, d2.top, d2.right, d2.bottom
    
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
    
    # # ----------------
    # # Using IOU
    # # ----------------
    # iou = computeIOU(detection, track.detections[-1])
    # #print('IOU( ' + detection.detection_id + ' , ' + track.track_id + ') = ' + str(iou))
    # if iou > iou_threshold: # This detection belongs to this tracker!!!
    #     track.update(detection) # add detection to track
    #     idxs_detections_to_remove.append(idx_detection)
    #     break # do not test this detection with any other track



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


def main():

    # Initialization
    cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    templates = []  # TODO: open templates saved on disk (add image format to .gitignore)


    # Execution
    while cap.isOpened():

        ret, frame = cap.read()
        if ret is False:
            break


        # Image processing
        img_bgr = copy.deepcopy(frame)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


        # Face detection
        faces = face_classifier.detectMultiScale(image=img_gray, scaleFactor=1.2, minNeighbors=4, minSize=(70,70))
        for (x,y,w,h) in faces:
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)


        # Visualization
        cv2.imshow('Camera', img_bgr)


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