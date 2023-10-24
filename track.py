#!/usr/bin/env python3

import cv2
import numpy as np

class Tracker():
    def __init__(self, img_original, name = 'Unknown'):
        self.img_original = img_original
        self.img_latest = img_original
        self.name = name
