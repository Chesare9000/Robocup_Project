import cv2
import numpy as np


def white_field_detector():


    # White Filter

    b_w = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # For the Field


    edge = cv2.Canny(b_w , 25, 75)

    cv2.imshow('field_edges', edge)

#end of field_detector()

def red_numbers_detector():

    # Red filter


    # Numbers detector

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0,70,50])
    upper_red_1 = np.array([10,255,255])

    lower_red_2 = np.array([170,70,255])
    upper_red_2 = np.array([180,50,255])

    mask = cv2.inRange(hsv, lower_red_1, upper_red_1) + cv2.inRange(hsv, lower_red_2, upper_red_2)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('red_binary_filter',mask)
    cv2.imshow('red_mask',res)



def calibrator():
    print("This is a test")

#main function
calibrator_flag = False ;

cap = cv2.VideoCapture(0) #global

while True:

    ret, frame = cap.read()  #frame is an uint8 numpy.ndarray

    frame = cv2.GaussianBlur(frame, (7, 7), 1.41) #smooth the Image

    white_field_detector()          # white filter + edges
    red_numbers_detector()         #red filter + number recognition

    if calibrator_flag == True:
        calibrator()

    #calibrator
    if cv2.waitKey(10) == ord('c'):
        False if calibrator_flag else True

    if cv2.waitKey(10) == ord('q'):  # 20 milisecond delay. press q to exit.
        break
