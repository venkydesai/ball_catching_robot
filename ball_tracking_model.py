import cv2
from ultralytics import YOLO
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
        
def main():
    model = YOLO("yolov8l-seg.pt")
    camera = cv2.VideoCapture(2)
    while True:
        ret, img = camera.read()
        if not ret:
            break
        # img = cv2.imread("../YOLO_image.jpg")
        # print(img.shape)
        results = model([img])[0]  # return a list of Results objects
        # del model
        # aruco_dict_type = aruco.DICT_5X5_100
        calibration_matrix_path = np.array([[636.99214003, 0, 326.0288417], [0, 637.21970636, 256.39394411],[0, 0, 1]])
        distortion_coefficients_path = np.array([[0.00916442, 0.16211854, -0.00320923, 0.00452386, -0.82387269]])
        # calibration_matrix_path =np.array([[982.47540652, 0, 339.77703514],[0, 989.06117495, 184.78593902],[0, 0, 1.0]])
        # distortion_coefficients_path = np.array([[  0.12331766, 0.3845027, -0.03356541, 0.011428, -1.20135501]])
        # Process results list
        objpoints = np.array([np.array([-0.03302, 0, 0]), np.array([0, -0.03302, 0]), np.array([0.03302, 0, 0]), np.array([0, 0.03302, 0])])
        # output = pose_esitmation(img, aruco_dict_type, calibration_matrix_path, distortion_coefficients_pa

        class_id=32
        if class_id in results.boxes.cls.numpy():
            position = (results.boxes.cls == 32).nonzero()
            pos=position.item()
            xywh=results.boxes.xywh[pos].numpy()
            x=int(xywh[0])
            y=int(xywh[1])
            wby2=int(xywh[2]/2)
            hby2=int(xywh[3]/2)
                
            top_left=(x-wby2,y-hby2)
            top_right=(x+wby2,y)
            bottom_left= (x,y+hby2)
            bottom_right=(x+wby2,y+hby2)
            # cv2.rectangle(img,top_left,bottom_right,(0,255,0),2)
            # cv2.imshow("image",img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            bottom=np.array([x,y+hby2])
            right=np.array([x+wby2,y])
            top=np.array([x,y-hby2])
            left=np.array([x-wby2,y])
        
            imgpoints = np.array([left, bottom, right, top])
            success, rvec, tvec = cv2.solvePnP(objpoints.astype('float32'), imgpoints.astype('float32'), calibration_matrix_path, distortion_coefficients_path)
                
            print("Success:", success)
            print("RVEC:", rvec)
            print("TVEC:", tvec)
            # cv2.rectangle(img,top_left,bottom_right,(0,255,0),2)
            cv2.imshow("image",img)
        
        if cv2.waitKey(1) & 0xFF is ord('q'):
            break
if __name__ == "__main__":
    main()