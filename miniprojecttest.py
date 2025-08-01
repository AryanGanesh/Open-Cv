import cv2
import numpy as np

def Colourandshapedetect(frame):
    #convert 2 hsv and set limits for the mask
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) 
    lowerhsv=np.array([7,31,86])
    higherhsv=np.array([28,104,255])

    mask=cv2.inRange(hsv, lowerhsv, higherhsv)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #preparing for contour detection
    imgray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

    #noise reduction steps
    blurred = cv2.bilateralFilter(imgray, 9, 75, 75)
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5,5), np.uint8)          
    thresh_eroded = cv2.erode(thresh_otsu, kernel, iterations=1)
    
    #contour finding
    contours, hierarchy = cv2.findContours(thresh_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contour_img = frame.copy()

    #finding largest contour and enclosing with MEC
    largest_contour = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 0, 255), 3)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        if radius > 5:
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(contour_img, center, radius, (0, 255, 0), 2)
    

    return res, imgray, blurred, thresh_eroded, contour_img



cap= cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not open")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    res,imgray,blurred, thresh_eroded, contour_img = Colourandshapedetect(frame)

    cv2.imshow("HSV result", res)
    cv2.imshow("Blurred",blurred)
    cv2.imshow("Grayed",imgray)
    cv2.imshow("eroded Threshold", thresh_eroded)
    cv2.imshow("Contours", contour_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()