import cv2
from deepface import DeepFace
import json

cap = cv2.VideoCapture(0)
attributes = ['age', 'gender', 'race', 'emotion']

while True:
    # Read the frame
    _, img = cap.read()
    demography = DeepFace.analyze(img, attributes)
    print(demography)
