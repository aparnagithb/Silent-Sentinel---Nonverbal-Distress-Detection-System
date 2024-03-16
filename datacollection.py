#import necessary modules
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


#set-up the camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
#set path for data to be stored
folder = "Custom_HandSign_DataSet/Thief"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    #check for if hand is detected
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        #crop out region of interest
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if the cropped region is valid
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            #resize the image while maintaining the ratio
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[hGap:hCal + hGap, :] = imgResize
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    #save the image when the key 's' is clicked
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
