import numpy as np
import cv2
import sys
from cv2 import aruco
# from utils import ARUCO_DICT
import argparse
import time

print(cv2.__version__)

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2_aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = aruco.detectMarkers(gray, cv2_aruco_dict,parameters=parameters,)
        # cameraMatrix=matrix_coefficients,
        # distCoeff=distortion_coefficients)

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.1, matrix_coefficients,
                                                                       distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 

            # Draw Axis
            print("TVEC")
            print(tvec)
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

    return frame


if __name__ == '__main__':


    aruco_dict_type = aruco.DICT_5X5_100

    calibration_matrix_path = np.array([[636.99214003, 0, 326.0288417], [0, 637.21970636, 256.39394411],[0, 0, 1]])
    # calibration_matrix_path = np.array([[525.43722984, 0, 304.94521649], [0, 523.38509633, 230.94930715],[0, 0, 1]])
    # calibration_matrix_path = np.array([[1.014e+03, 0.000e+00, 6.395e+02], [0.000e+00, 1.014e+03, 3.595e+02], [0.000e+00, 0.000e+00, 1.000e+00]])
    distortion_coefficients_path = np.array([[0.00916442, 0.16211854, -0.00320923, 0.00452386, -0.82387269]])
    # distortion_coefficients_path = np.array([[-0.00130835, 0.04182729, -0.00037811, -0.00882323, -0.1076405 ]])
    # distortion_coefficients_path = np.array([[-2.000e-01, 1.000e-02, 1.000e-03, 1.000e-03, 0.000e+00]])

    video = cv2.VideoCapture(2)
    time.sleep(2.0)
    try:
        while True:
            ret, frame = video.read()

            if not ret:
                break
            
            output = pose_esitmation(frame, aruco_dict_type, calibration_matrix_path, distortion_coefficients_path)

            cv2.imshow('Estimated Pose', output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    except:

        video.release()
        cv2.destroyAllWindows()
        sys.exit(1)