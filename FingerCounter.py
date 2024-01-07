import mediapipe as mp
import cv2
import time
import os
import HandTrackerModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "" # Have To Write Change This To Image Path
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)

prevTime = 0

detector = htm.handDetector(detectionConf=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmlist = detector.find_position(img, draw=False)
    if len(lmlist) != 0:
        fingers = []
        if lmlist[tipIds[0]][1] < lmlist[tipIds[0] + 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)

        h, w, c = overlayList[0].shape
        img[0:h, 0:w] = cv2.resize(overlayList[totalFingers-1], (w, h))




    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (600, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
