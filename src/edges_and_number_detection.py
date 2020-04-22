#ROBOCUP PROJECT
#CESAR GONZALEZ CRUZ
#Student ID: 0414932


import cv2
import numpy as np
import time
import json


def white_field_detector():


    # White Filter

    b_w = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # For the Field


    edge = cv2.Canny(b_w , 25, 75)

    cv2.imshow('field_edges', edge)

#end of field_detector()

def red_numbers_detector():

    global refresh , contour_thresh , contour_max_thresh
    global contour_min_area , contour_max_area

    if refresh == True:

        #Getting Bounding Boxes


        #lower_red_1 = np.array([0,70,50])
        #upper_red_1 = np.array([10,255,255])

        #lower_red_2 = np.array([170,70,255])
        #upper_red_2 = np.array([180,50,255])

        #mask = cv2.inRange(hsv, lower_red_1, upper_red_1) + cv2.inRange(hsv, lower_red_2, upper_red_2)
        #res = cv2.bitwise_and(frame,frame, mask= mask)

        #retrieving blue filter
        cv2.namedWindow('Blue Mask')
        cv2.namedWindow('ROI Contours')

        cv2.namedWindow('Red Mask')
        cv2.namedWindow('Number Contours')


        contour_max_thresh = parameters.get('contour_max_thresh')
        contour_thresh = parameters.get('contour_thresh')

        contour_min_area = parameters.get('contour_min_area')
        contour_max_area = parameters.get('contour_max_area')

        # show thresholded images
        cv2.createTrackbar('ROI_THRESHOLD','ROI Contours', contour_thresh, contour_max_thresh,callback)

        cv2.createTrackbar('MINIMUM_AREA','ROI Contours', contour_min_area, contour_max_area,callback)

        refresh = False

#to update

    parameters['contour_thresh'] = cv2.getTrackbarPos('ROI_THRESHOLD', 'ROI Contours')
    contour_thresh = parameters.get('contour_thresh')

    parameters['contour_min_area'] = cv2.getTrackbarPos('MINIMUM_AREA', 'ROI Contours')
    contour_min_area = parameters.get('contour_min_area')

    blue_lower_hsv = np.array([blue_h_min, blue_s_min, blue_v_min])
    blue_higher_hsv = np.array([blue_h_max, blue_s_max, blue_v_max])

    blue_mask = cv2.inRange(hsv, blue_lower_hsv, blue_higher_hsv)
    blue_res = cv2.bitwise_and(frame, frame, mask=blue_mask)

    red_lower_hsv = np.array([red_h_min, red_s_min, red_v_min])
    red_higher_hsv = np.array([red_h_max, red_s_max, red_v_max])

    red_mask = cv2.inRange(hsv, red_lower_hsv, red_higher_hsv)
    red_res = cv2.bitwise_and(frame, frame, mask=red_mask)


    src_gray_for_blue = cv2.cvtColor(blue_res, cv2.COLOR_BGR2GRAY)
    src_gray_for_blue = cv2.blur(src_gray_for_blue, (3,3))

    src_gray_for_red = cv2.cvtColor(red_res, cv2.COLOR_BGR2GRAY)
    src_gray_for_red = cv2.blur(src_gray_for_red, (3,3))


    canny_output_for_blue = cv2.Canny(src_gray_for_blue, contour_thresh, contour_thresh * 2)
    canny_output_for_red = cv2.Canny(src_gray_for_red, contour_thresh, contour_thresh * 2)

    _, blue_contours, _ = cv2.findContours(canny_output_for_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, red_contours, _ = cv2.findContours(canny_output_for_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_contours_poly = [None]*len(blue_contours)
    blue_boundRect = [None]*len(blue_contours)
    blue_centers = [None]*len(blue_contours)
    blue_radius = [None]*len(blue_contours)

    red_contours_poly = [None]*len(red_contours)
    red_boundRect = [None]*len(red_contours)
    red_centers = [None]*len(red_contours)
    red_radius = [None]*len(red_contours)


    for i, c in enumerate(blue_contours):
        blue_contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        blue_boundRect[i] = cv2.boundingRect(blue_contours_poly[i])
        blue_centers[i], blue_radius[i] = cv2.minEnclosingCircle(blue_contours_poly[i])

    for i, c in enumerate(red_contours):
        red_contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        red_boundRect[i] = cv2.boundingRect(red_contours_poly[i])
        red_centers[i], red_radius[i] = cv2.minEnclosingCircle(red_contours_poly[i])

    blue_drawing = np.zeros((canny_output_for_blue.shape[0], canny_output_for_blue.shape[1], 3), dtype=np.uint8)
    red_drawing = np.zeros((canny_output_for_red.shape[0], canny_output_for_red.shape[1], 3), dtype=np.uint8)

    blue_color = (255, 0 , 0)
    red_color = (0,0,255)

    blue_cell_counter = 0
    red_cell_counter = 0

    for i in range(len(blue_contours)):
        if (cv2.contourArea(blue_contours[i]) > contour_min_area):
            cv2.drawContours(blue_drawing, blue_contours_poly, i, blue_color)
            #cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
             # (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            #cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
            blue_cell_counter+=1

    for i in range(len(red_contours)):
        if (cv2.contourArea(red_contours[i]) > contour_min_area):
            cv2.drawContours(red_drawing, red_contours_poly, i, red_color)
            #cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
             # (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            #cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
            red_cell_counter+=1

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
#   bottomLeftCornerOfText = (10,450) Depends on the webcam resolution
    fontScale              = 1
    fontColor              = (0,255,0)
    lineType               = 2

    cv2.putText(blue_drawing,'Occupied cells: ' + str(blue_cell_counter) , bottomLeftCornerOfText,font,fontScale,fontColor,lineType)

    cv2.putText(red_drawing,'Empty cells: ' + str(red_cell_counter) , bottomLeftCornerOfText,font,fontScale,fontColor,lineType)

    cv2.imshow('Blue Mask', blue_res)
    cv2.imshow('ROI Contours', blue_drawing)

    cv2.imshow('Red Mask', red_res)
    cv2.imshow('Number Contours', red_drawing)

    #cv2.imshow('red_binary_filter',mask)
    #cv2.imshow('red_mask',res)



def calibrator():


    global calibrator_state

    global min , max , max_h

    global red_h_min, green_h_min, blue_h_min
    global red_h_max ,green_h_max, blue_h_max

    global red_s_min, green_s_min, blue_s_min
    global red_s_max, green_s_max, blue_s_max

    global red_v_min, green_v_min, blue_v_min
    global red_v_max, green_v_max, blue_v_max

    global parameters

    global   red_lower_hsv ,   red_higher_hsv
    global green_lower_hsv , green_higher_hsv
    global  blue_lower_hsv ,  blue_higher_hsv

    global   red_mask ,   red_res
    global green_mask , green_res
    global  blue_mask ,  blue_res


    if calibrator_state == False:

        with open('parameters.json', 'r') as fp:
            parameters = json.load(fp)


        min   = parameters.get('min')
        max   = parameters.get('max')
        max_h = parameters.get('max_h')

        red_h_min = parameters.get('red_h_min')
        red_h_max = parameters.get('red_h_max')

        red_s_min = parameters.get('red_s_min')
        red_s_max = parameters.get('red_s_max')

        red_v_min = parameters.get('red_v_min')
        red_v_max = parameters.get('red_v_max')


        green_h_min = parameters.get('green_h_min')
        green_h_max = parameters.get('green_h_max')

        green_s_min = parameters.get('green_s_min')
        green_s_max = parameters.get('green_s_max')

        green_v_min = parameters.get('green_v_min')
        green_v_max = parameters.get('green_v_max')


        blue_h_min = parameters.get('blue_h_min')
        blue_h_max = parameters.get('blue_h_max')

        blue_s_min = parameters.get('blue_s_min')
        blue_s_max = parameters.get('blue_s_max')

        blue_v_min = parameters.get('blue_v_min')
        blue_v_max = parameters.get('blue_v_max')





        #mover a main . icluir tambien los de arriba que borre



        cv2.namedWindow('HSV Calibrator for RED')

        cv2.createTrackbar('RED_H_MIN','HSV Calibrator for RED',red_h_min,max_h,callback)
        cv2.createTrackbar('RED_H_MAX','HSV Calibrator for RED',red_h_max,max_h,callback)

        cv2.createTrackbar('RED_S_MIN','HSV Calibrator for RED',red_s_min,max,callback)
        cv2.createTrackbar('RED_S_MAX','HSV Calibrator for RED',red_s_max,max,callback)

        cv2.createTrackbar('RED_V_MIN','HSV Calibrator for RED',red_v_min,max,callback)
        cv2.createTrackbar('RED_V_MAX','HSV Calibrator for RED',red_v_max,max,callback)

        cv2.namedWindow('HSV Calibrator for GREEN')

        cv2.createTrackbar('GREEN_H_MIN','HSV Calibrator for GREEN',green_h_min,max_h,callback)
        cv2.createTrackbar('GREEN_H_MAX','HSV Calibrator for GREEN',green_h_max,max_h,callback)

        cv2.createTrackbar('GREEN_S_MIN','HSV Calibrator for GREEN',green_s_min,max,callback)
        cv2.createTrackbar('GREEN_S_MAX','HSV Calibrator for GREEN',green_s_max,max,callback)

        cv2.createTrackbar('GREEN_V_MIN','HSV Calibrator for GREEN',green_v_min,max,callback)
        cv2.createTrackbar('GREEN_V_MAX','HSV Calibrator for GREEN',green_v_max,max,callback)

        cv2.namedWindow('HSV Calibrator for BLUE')

        cv2.createTrackbar('BLUE_H_MIN','HSV Calibrator for BLUE',blue_h_min,max_h,callback)
        cv2.createTrackbar('BLUE_H_MAX','HSV Calibrator for BLUE',blue_h_max,max_h,callback)

        cv2.createTrackbar('BLUE_S_MIN','HSV Calibrator for BLUE',blue_s_min,max,callback)
        cv2.createTrackbar('BLUE_S_MAX','HSV Calibrator for BLUE',blue_s_max,max,callback)

        cv2.createTrackbar('BLUE_V_MIN','HSV Calibrator for BLUE',blue_v_min,max,callback)
        cv2.createTrackbar('BLUE_V_MAX','HSV Calibrator for BLUE',blue_v_max,max,callback)

        calibrator_state = True

    #to update

    parameters['red_h_min'] = cv2.getTrackbarPos('RED_H_MIN', 'HSV Calibrator for RED')
    parameters['red_h_max'] = cv2.getTrackbarPos('RED_H_MAX', 'HSV Calibrator for RED')

    parameters['red_s_min'] = cv2.getTrackbarPos('RED_S_MIN', 'HSV Calibrator for RED')
    parameters['red_s_max'] = cv2.getTrackbarPos('RED_S_MAX', 'HSV Calibrator for RED')

    parameters['red_v_min'] = cv2.getTrackbarPos('RED_V_MIN', 'HSV Calibrator for RED')
    parameters['red_v_max'] = cv2.getTrackbarPos('RED_V_MAX', 'HSV Calibrator for RED')

    parameters['green_h_min'] = cv2.getTrackbarPos('GREEN_H_MIN', 'HSV Calibrator for GREEN')
    parameters['green_h_max'] = cv2.getTrackbarPos('GREEN_H_MAX', 'HSV Calibrator for GREEN')

    parameters['green_s_min'] = cv2.getTrackbarPos('GREEN_S_MIN', 'HSV Calibrator for GREEN')
    parameters['green_s_max'] = cv2.getTrackbarPos('GREEN_S_MAX', 'HSV Calibrator for GREEN')

    parameters['green_v_min'] = cv2.getTrackbarPos('GREEN_V_MIN', 'HSV Calibrator for GREEN')
    parameters['green_v_max'] = cv2.getTrackbarPos('GREEN_V_MAX', 'HSV Calibrator for GREEN')

    parameters['blue_h_min'] = cv2.getTrackbarPos('BLUE_H_MIN', 'HSV Calibrator for BLUE')
    parameters['blue_h_max'] = cv2.getTrackbarPos('BLUE_H_MAX', 'HSV Calibrator for BLUE')

    parameters['blue_s_min'] = cv2.getTrackbarPos('BLUE_S_MIN', 'HSV Calibrator for BLUE')
    parameters['blue_s_max'] = cv2.getTrackbarPos('BLUE_S_MAX', 'HSV Calibrator for BLUE')

    parameters['blue_v_min'] = cv2.getTrackbarPos('BLUE_V_MIN', 'HSV Calibrator for BLUE')
    parameters['blue_v_max'] = cv2.getTrackbarPos('BLUE_V_MAX', 'HSV Calibrator for BLUE')

    #We could also just save the vars on the dict at the end.... maybe for v2 ;)

    red_h_min = parameters.get('red_h_min')
    red_h_max = parameters.get('red_h_max')

    red_s_min = parameters.get('red_s_min')
    red_s_max = parameters.get('red_s_max')

    red_v_min = parameters.get('red_v_min')
    red_v_max = parameters.get('red_v_max')


    green_h_min = parameters.get('green_h_min')
    green_h_max = parameters.get('green_h_max')

    green_s_min = parameters.get('green_s_min')
    green_s_max = parameters.get('green_s_max')

    green_v_min = parameters.get('green_v_min')
    green_v_max = parameters.get('green_v_max')


    blue_h_min = parameters.get('blue_h_min')
    blue_h_max = parameters.get('blue_h_max')

    blue_s_min = parameters.get('blue_s_min')
    blue_s_max = parameters.get('blue_s_max')

    blue_v_min = parameters.get('blue_v_min')
    blue_v_max = parameters.get('blue_v_max')

    #end of to update

    red_lower_hsv = np.array([red_h_min, red_s_min, red_v_min])
    red_higher_hsv = np.array([red_h_max, red_s_max, red_v_max])

    green_lower_hsv = np.array([green_h_min, green_s_min, green_v_min])
    green_higher_hsv = np.array([green_h_max, green_s_max, green_v_max])

    blue_lower_hsv = np.array([blue_h_min, blue_s_min, blue_v_min])
    blue_higher_hsv = np.array([blue_h_max, blue_s_max, blue_v_max])


    red_mask = cv2.inRange(hsv, red_lower_hsv, red_higher_hsv)
    green_mask = cv2.inRange(hsv, green_lower_hsv, green_higher_hsv)
    blue_mask = cv2.inRange(hsv, blue_lower_hsv, blue_higher_hsv)


    red_res = cv2.bitwise_and(frame, frame, mask=red_mask)
    green_res = cv2.bitwise_and(frame, frame, mask=green_mask)
    blue_res = cv2.bitwise_and(frame, frame, mask=blue_mask)



    # show thresholded images
    cv2.imshow('Original Image', frame)
    cv2.imshow('HSV Calibrator for RED', red_res)
    cv2.imshow('HSV Calibrator for GREEN', green_res)
    cv2.imshow('HSV Calibrator for BLUE', blue_res)

    k = cv2.waitKey(10) & 0xFF # large wait time to remove freezing


#dummy iterator fot the trackbars
def callback(x):
    pass

#END OF FUNCTIONS
#----------------------------------------

calibrator_flag = False # to start it
calibrator_state = False # to know the status (starting vs started)

refresh = True

#main function



cap = cv2.VideoCapture(0) #0 is default but 2 is my ext. webcam

#Just once to populate the Dictionary----------------------------------------------
ret, frame = cap.read()  #frame is an uint8 numpy.ndarray
frame = cv2.GaussianBlur(frame, (7, 7), 1.41) #smooth the Image
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
calibrator()
#----------------------------------------------------------------------------

while True:

    ret, frame = cap.read()  #frame is an uint8 numpy.ndarray

    frame = cv2.GaussianBlur(frame, (7, 7), 1.41) #smooth the Image

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if calibrator_flag == True:
        calibrator()
    else:
        #later kill each one with diff keys
        cv2.destroyWindow("HSV Calibrator for RED")
        cv2.destroyWindow("HSV Calibrator for GREEN")
        cv2.destroyWindow("HSV Calibrator for BLUE")
        cv2.destroyWindow("Original Image")

    #always Running
    white_field_detector()          # white filter + edges
    red_numbers_detector()         #red filter + number recognition



    if cv2.waitKey(10) == ord('c'):
        if calibrator_flag == False:
            calibrator_state = False
            calibrator_flag  = True
            refresh = True
        else:
            calibrator_flag = False
            refresh = True
            with open('parameters.json', 'w') as fp:
                json.dump(parameters, fp, indent=4)

    elif cv2.waitKey(10) == ord('q'):  #  milisecond delay. press q to exit.
        with open('parameters.json', 'w') as fp:
            json.dump(parameters, fp, indent=4)
        break
