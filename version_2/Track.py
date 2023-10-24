import cv2
import numpy as np


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
    
class Detection():
    def __init__(self,left,right,top,bottom,id,stamp,image):
        self.detection_id = id
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.time_stamp = stamp
        self.img_detection = image[self.top:self.bottom,self.left:self.right]

    def draw(self,img,color,draw_position='bottom',text=None):
        start_point = (self.left,self.top)
        end_point = (self.right,self.bottom)

        # draw rectangle around face
        cv2.rectangle(img,start_point,end_point,color,2)

        # identify face detected
        if text is None:
            text = 'Det ' + self.detection_id

        if draw_position == 'bottom':
            position = (self.left,self.bottom+30)
        else:
            position = (self.left,self.top-10)  

        cv2.putText(img, text,position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        

class Track:
    def __init__(self,id,detection,color = (255,0,0)): # definition of class constructor
        self.track_id = id
        self.detections = [detection]
        self.active = True
        self.color = color

        
    def draw(self,image):
        
        #Draw only last detection
        self.detections[-1].draw(image,self.color, text='Person: ' + self.track_id,draw_position='top')



        #left, right, top, bottom = self.detections[-1]

        # cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),2)
        # image = cv2.putText(image,'Person: ' + str(self._trackid),(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2,cv2.LINE_AA)

    def track():
        print('a')
        # function to do the template mactching using self.detections[-1].img_detection !!!
        # use img_gray to match 



    def update(self,detection):
        
        self.detections.append(detection)

