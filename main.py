#!/usr/bin/env python3

import cv2
import numpy as np
import copy
import tkinter as tk
from tkinter import simpledialog


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