import cv2
import numpy as np

img = cv2.imread("images/original.jpg", cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 50, 150, apertureSize=3)
cv2.imwrite("images/tepi.jpg", edges)

kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)
cv2.imwrite("images/dilation.jpg", dilated_edges)