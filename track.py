#!/usr/bin/env python3

import cv2
import numpy as np
import time

class Tracker():
    def __init__(self, img_original, name):

        self.name = name
        self.img_original = img_original
        self.img_latest = img_original
        self.last_face_timestamp = time.time()

    def reset(self):
        self.img_latest = self.img_original
