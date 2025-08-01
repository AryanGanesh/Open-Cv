import cv2
import numpy as np

def nothing(x):
    pass

LOW_H_default = 0
HIGH_H_default = 179
LOW_S_default = 0
HIGH_S_default = 255
LOW_V_default = 0
HIGH_V_default = 255

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not open!!")
    exit()

cv2.namedWindow('HSV Threshold')
cv2.createTrackbar('Low H','HSV Threshold',LOW_H_default,179,nothing)
cv2.createTrackbar('High H','HSV Threshold',HIGH_H_default,179,nothing)
cv2.createTrackbar('Low S','HSV Threshold',LOW_S_default,255,nothing)
cv2.createTrackbar('High S','HSV Threshold',HIGH_S_default,255,nothing)
cv2.createTrackbar('Low V','HSV Threshold',LOW_V_default,255,nothing)
cv2.createTrackbar('High V','HSV Threshold',HIGH_V_default,255,nothing)
cv2.createTrackbar('Reset All','HSV Threshold',0,1,nothing)

# Define a kernel (structuring element) for erosion:
kernel = np.ones((5,5), np.uint8)  # you can adjust kernel size

print("Use the trackbars to adjust HSV values and find your ideal threshold range.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed, exiting.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Reset all trackbars if Reset All > 0:
    if cv2.getTrackbarPos('Reset All','HSV Threshold') > 0:
        cv2.setTrackbarPos('Low H','HSV Threshold',LOW_H_default)
        cv2.setTrackbarPos('High H','HSV Threshold',HIGH_H_default)
        cv2.setTrackbarPos('Low S','HSV Threshold',LOW_S_default)
        cv2.setTrackbarPos('High S','HSV Threshold',HIGH_S_default)
        cv2.setTrackbarPos('Low V','HSV Threshold',LOW_V_default)
        cv2.setTrackbarPos('High V','HSV Threshold',HIGH_V_default)
        cv2.setTrackbarPos('Reset All','HSV Threshold',0)

    low_h = cv2.getTrackbarPos('Low H','HSV Threshold')
    high_h = cv2.getTrackbarPos('High H','HSV Threshold')
    low_s = cv2.getTrackbarPos('Low S','HSV Threshold')
    high_s = cv2.getTrackbarPos('High S','HSV Threshold')
    low_v = cv2.getTrackbarPos('Low V','HSV Threshold')
    high_v = cv2.getTrackbarPos('High V','HSV Threshold')

    lower_hsv = np.array([low_h, low_s, low_v])
    upper_hsv = np.array([high_h, high_s, high_v])

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # --- Erode the mask to reduce noise ---
    mask_eroded = cv2.erode(mask, kernel, iterations=1)

    # Apply masked bitwise_and using eroded mask instead of raw mask
    result = cv2.bitwise_and(frame, frame, mask=mask_eroded)

    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Mask Eroded', mask_eroded)  # To see noise removed after erosion
    cv2.imshow('Result', result)

    print_str = f"Lower HSV: [{low_h}, {low_s}, {low_v}]  Upper HSV: [{high_h}, {high_s}, {high_v}]"
    frame_disp = frame.copy()
    cv2.putText(frame_disp, print_str, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow('HSV Threshold', frame_disp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Final HSV range:')
        print(f'Lower HSV: [{low_h}, {low_s}, {low_v}]')
        print(f'Upper HSV: [{high_h}, {high_s}, {high_v}]')
        break

cap.release()
cv2.destroyAllWindows()
