from PIL import Image
import cv2
import numpy as np
import os

# Konversi ke grayscale
img_gray = cv2.imread("images/original.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite('images/grayscale.jpg', img_gray)

# Thresholding
img_gray = cv2.imread('images/grayscale.jpg', cv2.IMREAD_GRAYSCALE)
threshold_value = 127
ret, thresholded_img = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
cv2.imwrite('images/thresholded_image.jpg', thresholded_img)

# Deteksi tepi
edges = cv2.Canny(thresholded_img, 50, 150, apertureSize=3)
cv2.imwrite("images/edges.jpg", edges)

# Operasi closing
kernel = np.ones((12,12),np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('images/closing.jpg', closing)

# Kompresi
closing_image = Image.fromarray(closing)
compressed_image = closing_image.resize((closing_image.width // 2, closing_image.height // 2))
compressed_image.save("images/compressed_image.jpg")

original_size = os.path.getsize('images/closing.jpg')
compressed_size = os.path.getsize('images/compressed_image.jpg')

compression_ratio_kb = (original_size / 1024) / (compressed_size / 1024)

print(f"Closing image size: {original_size/1024:.2f} KB")
print(f"Compressed image size: {compressed_size/1024:.2f} KB")
print(f"Compression ratio: {compression_ratio_kb:.2f}")

print('\nComparison:\n')

original_size = os.path.getsize('images/original.jpg')
print(f"Original image size: {original_size/1024:.2f} KB")

gray_size = os.path.getsize('images/grayscale.jpg')
print(f"Grayscale image size: {gray_size/1024:.2f} KB")

gray_size = os.path.getsize('images/thresholded_image.jpg')
print(f"Thresholded image size: {gray_size/1024:.2f} KB")

edges_size = os.path.getsize('images/edges.jpg')
print(f"Edges image size: {edges_size/1024:.2f} KB")

dilated_size = os.path.getsize('images/closing.jpg')
print(f"Closing image size: {dilated_size/1024:.2f} KB")

compressed_size = os.path.getsize('images/compressed_image.jpg')
print(f"Compressed image size: {compressed_size/1024:.2f} KB")