import numpy as np
import matplotlib.pyplot as plt
import cv2


height,width=200,300
gradient = np.zeros((height,width),dtype=np.uint8)
for i in range(height):
    gradient[i,:]=int((i/height)*225)

cv2.imshow('1. Vertical Gradient', gradient)
cv2.waitKey(0)

circle_img= np.zeros((200,200),dtype=np.uint8)
cv2.circle(circle_img,(100,100),50,255,-1)

cv2.imshow('2. Filled Circle', circle_img)
cv2.waitKey(0)

img=cv2.imread(r'c:\Users\aryan\OneDrive - Thapar University\Pictures\Screenshots\Screenshot 2025-10-07 104310.png')

# Check if image was loaded
if img is None:
    print("Error: Image not loaded. Check file path.")
    cv2.destroyAllWindows()
    exit()

img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

print(f'Shape: {img_rgb.shape}')
print(f"Data type: {img_rgb.dtype}")
print(f"Min/Max values: {img_rgb.min()}, {img_rgb.max()}")

pixel = img_rgb[100, 150] 
print(f"Pixel RGB values: {pixel}")

img_copy = img_rgb.copy()
img_copy[50:100, 50:100] = [255, 0, 0] # Red square

# Since cv2.imshow expects BGR for color, convert back from RGB for display
img_display_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
cv2.imshow('3. Image with Red Square', img_display_bgr)
cv2.waitKey(0)


b, g, r = cv2.split(img)

cv2.imshow('4. Blue Channel', b)
cv2.waitKey(0)

blur_kernel = np.ones((5,5), np.float32) / 25
blurred = cv2.filter2D(img, -1, blur_kernel)

cv2.imshow('5. Blurred Image', blurred)
cv2.waitKey(0)

sobel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
edges_x = cv2.filter2D(img, -1, sobel_x)

cv2.imshow('6. X-Edges (Sobel)', edges_x)
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('7. Simple Binary Threshold', binary)
cv2.waitKey(0)

adaptive = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

cv2.imshow('8. Adaptive Threshold', adaptive)
cv2.waitKey(0)

kernel = np.ones((5,5), np.uint8)

# Erosion: shrinks objects
eroded = cv2.erode(binary, kernel, iterations=1)

cv2.imshow('9. Erosion', eroded)
cv2.waitKey(0)

# Dilation: grows objects
dilated = cv2.dilate(binary, kernel, iterations=1)

cv2.imshow('10. Dilation', dilated)
cv2.waitKey(0)

# Opening: erosion followed by dilation (removes noise)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

cv2.imshow('11. Opening', opening)
cv2.waitKey(0)

# Closing: dilation followed by erosion (fills holes)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

cv2.imshow('12. Closing', closing)
cv2.waitKey(0)

cv2.destroyAllWindows()