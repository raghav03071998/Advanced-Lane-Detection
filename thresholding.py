import cv2
import numpy as np
import pandas as pd


def binary_image(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    sobel_x=cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 5)
    ret,thresh1 = cv2.threshold(sobel_x,250,255,cv2.THRESH_BINARY)
    return [ret,thresh1]
    
