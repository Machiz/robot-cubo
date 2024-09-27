import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(1)
detector = cvzone.HandDetector(maxHands = 1, detectionCon = 0.7)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist, = detector.findPosition(img)
    if lmlist:
        fingers = detector.fingersUP()
        print(fingers)
    cv2.imshow('image', img)
    cv2.waitKey(1)