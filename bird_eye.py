import numpy as np
import pandas as pd
import cv2
import pickle


#GETTING THE CORRECTION COEFFICIENTS FROM CAMERA CALIBATION#
calibration_data = pickle.load(open("calibration_data.p", "rb" ))
matrix = calibration_data['camera_matrix']
dist_coef = calibration_data['distortion_coefficient']


source_points = np.float32([(580, 460), (205, 720), (1110, 720), (703, 460)])#THE ORIGINAL 4 POINTS#
dest_points = np.float32([(320, 0), (320, 720), (960, 720), (960, 0)])#THE POSITION THEY SHOULD BE IN #


#GETTING THE PERSPECTIVE TRANSFORM#
warp_matrix = cv2.getPerspectiveTransform(source_points,dest_points)

###ONE FUNCTION FOR ALL STEPS###
def birdeye(img,warp_matrix,matrix,dist_coef):
    undistorted_image=cv2.undistort(img,matrix,dist_coef, None, matrix)
    warped_image=cv2.warpPerspective(img, warp_matrix, (img.shape[1],img.shape[0]), flags = cv2.INTER_LINEAR)
    return warped_image
    
    
