import cv2 
import numpy as np


source_points = np.float32([(580, 460), (205, 720), (1110, 720), (703, 460)])
destination_points = np.float32([(320, 0), (320, 720), (960, 720), (960, 0)])


#PROJECTING THE LANE LINE ON THE IMAGE#
def project(ground_image, sky_lane, left_fit, right_fit):


    z = np.zeros_like(sky_lane)
    sky_lane = np.dstack((z, z, z))

    kl, kr = left_fit, right_fit
    h = sky_lane.shape[0]
    ys = np.linspace(0, h - 1, h)
    lxs = kl[0] * (ys**2) + kl[1]* ys +  kl[2]
    rxs = kr[0] * (ys**2) + kr[1]* ys +  kr[2]
    
    pts_left = np.array([np.transpose(np.vstack([lxs, ys]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rxs, ys])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(sky_lane, np.int_(pts), (0,255,0))
    
    shape = (sky_lane.shape[1], sky_lane.shape[0])
    ground_lane = cv2.warpPerspective(sky_lane, cv2.getPerspectiveTransform(destination_points,source_points), shape)
    ground_lane=np.uint8(ground_lane)
    result = cv2.addWeighted(ground_image, 1, ground_lane, 0.3, 0)
    return result
