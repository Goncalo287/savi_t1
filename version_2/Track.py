import cv2
import numpy as np


def computeIOU(d1, d2):
    x,y,w,h = d2
    # box1 and box2 should be in the format (x1, y1, x2, y2)
    x1_1, y1_1, x2_1, y2_1 = d1
    x1_2, y1_2, x2_2, y2_2 = x,y,x+w,y+h
    
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

class Tracker():
    def __init__(self, img_original,img_last,name,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.img_original = img_original
        self.name = name
        self.img_last=img_last

    def draw(self,image):
        start_point = (self.x,self.y)
        end_point = (self.x+self.w,self.y+self.h)
        cv2.rectangle(image,start_point,end_point,(0,0,255),2)
