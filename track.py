#!/usr/bin/env python3

import cv2
import numpy as np

class Tracker():
    def __init__(self, tracker, initBB, image, name):
        self.tracker = tracker
        self.initBB = initBB
        self.image = image
        self.name = name

    