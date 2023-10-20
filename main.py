#!/usr/bin/env python3

import cv2
import numpy as np
import copy



def main():

    # Initialization
    cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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


    # Destroy cv2 windows
    cap.release()
    cv2.destroyAllWindows()
        


if __name__ == '__main__':
    main()