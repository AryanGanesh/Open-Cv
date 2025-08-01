import cv2
import numpy as np


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # set height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for ball (bright green/yellow)
    lower_hsv = np.array([29, 86, 6])
    upper_hsv = np.array([64, 255, 255])

    # Threshold the HSV image to get only tennis ball colors
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Morphological operations to remove small noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours on the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None

    if len(contours) > 0:
        # Find the largest contour assuming it's the ball
        c = max(contours, key=cv2.contourArea)

        # Minimum enclosing circle around the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Calculate centroid for the contour
        M = cv2.moments(c)
        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (int(x), int(y))

        # Only proceed if the radius is large enough to filter noise
        if radius > 10:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 3)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Show the frame with tracking
    cv2.imshow("Tennis Ball Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
