import cv2
import time
import os
import hand_tracker as ht

width, height = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)


detector = ht.hand_detector(detectionCon=0.75)

# 4 = thumb, 8 = pointer finger, ... 20 = pinky finger
fingertip_ids = [4, 8, 12, 16, 20]

while True:
    _, image = cap.read()
    image = cv2.flip(image, 1)
    image = detector.find_hands(image)
    # https://google.github.io/mediapipe/solutions/hands.html
    landmark_list = detector.find_position(image, draw=False)
    #print(landmark_list)
    total_fingers = 0

    if len(landmark_list) != 0:
        fingers = []

        # For the Thumb
        if landmark_list[fingertip_ids[0]][1] < landmark_list[fingertip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # For the other Fingers
        for id in range(1,5):
            if landmark_list[fingertip_ids[id]][2] < landmark_list[fingertip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)
        total_fingers = fingers.count(1)
        #print(total_fingers)



    cv2.putText(image, f"Anzahl: {int(total_fingers)}", (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", image)
    cv2.waitKey(1)

    # Close Window
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

        

cv2.destroyAllWindows()
