import cv2
import numpy as np
import time

last_print_time = 0 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open Web cam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    half = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
    half=cv2.flip(half,1)
    gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(25, 25))
    #edged = cv2.Canny(gray, 30, 200)
    #contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(half, contours, -1, (0, 0, 0), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(half, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center_x = x + w // 2
        center_y = y + h // 2
        print(f"center_x={center_x}, center_y={center_y}")
        #print(f"x={x},y={y}")
        #   current_time = time.time()
        #if current_time - last_print_time > 1:  # 1 second has passed
        #    print(f"x={center_x},y={center_y}")
        #    last_print_time = current_time
    cv2.imshow('Face Detection', half)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
