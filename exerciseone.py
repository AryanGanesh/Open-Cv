import numpy as np
import matplotlib.pyplot as plt
import cv2


height,width=200,300
gradient = np.zeros((height,width),dtype=np.uint8)
for i in range(height):
    gradient[i,:]=int((i/height)*225)

circle_img= np.zeros((200,200),dtype=np.uint8)
cv2.circle(circle_img,(100,100),50,255,-1)

img=cv2.imread(r'c:\Users\aryan\OneDrive - Thapar University\Pictures\Screenshots\Screenshot 2025-10-07 104310.png')
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

print(f'Shape: {img_rgb.shape}')
print(f"Data type: {img_rgb.dtype}")
print(f"Min/Max values: {img_rgb.min()}, {img_rgb.max()}")

pixel = img_rgb[100, 150] 
print(f"Pixel RGB values: {pixel}")

img_copy = img_rgb.copy()
img_copy[50:100, 50:100] = [255, 0, 0]  # Red square

b, g, r = cv2.split(img)