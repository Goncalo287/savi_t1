#!/usr/bin/env python3

import cv2
import numpy as np
import copy



def main():

    # Initialization
    cap = cv2.VideoCapture(0)


    # Execution
    while cap.isOpened():

        ret, frame = cap.read()
        if ret is False:
            break


        # Image processing
        img_bgr = copy.deepcopy(frame)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


        # Visualization
        cv2.imshow('Camera', img_gray)


        # Keyboard inputs
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:    # Q or ESC to exit
            break


    # Destroy cv2 windows
    cap.release()
    cv2.destroyAllWindows()
        


if __name__ == '__main__':
    main()