from PIL import Image
import cv2
import numpy as np
import os

# Konversi ke grayscale
img_gray = cv2.imread("images/original.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite('images/grayscale.jpg', img_gray)

# Deteksi tepi
edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
cv2.imwrite("images/edges.jpg", edges)

# Operasi dilasi
kernel = [
    [0,0,0],
    [1,1,1],
    [0,1,0]
  ]
kernel = np.array(kernel, dtype=np.uint8)

dilated_edges = cv2.dilate(edges, kernel, iterations=1)
cv2.imwrite("images/dilation.jpg", dilated_edges)

# Kompresi
dilated_image = Image.fromarray(dilated_edges)
compressed_image = dilated_image.resize((dilated_image.width // 2, dilated_image.height // 2))
compressed_image.save("images/compressed_dilated_edges.jpg")

original_size = os.path.getsize('images/dilation.jpg')
compressed_size = os.path.getsize('images/compressed_dilated_edges.jpg')

compression_ratio_kb = (original_size / 1024) / (compressed_size / 1024)

print(f"Dilated image size: {original_size/1024:.2f} KB")
print(f"Compressed image size: {compressed_size/1024:.2f} KB")
print(f"Compression ratio: {compression_ratio_kb:.2f}")

print('\nComparison:\n')

original_size = os.path.getsize('images/original.jpg')
print(f"Original image size: {original_size/1024:.2f} KB")

gray_size = os.path.getsize('images/grayscale.jpg')
print(f"Grayscale image size: {gray_size/1024:.2f} KB")

edges_size = os.path.getsize('images/edges.jpg')
print(f"Edges image size: {edges_size/1024:.2f} KB")

dilated_size = os.path.getsize('images/dilation.jpg')
print(f"Dilated image size: {dilated_size/1024:.2f} KB")

compressed_size = os.path.getsize('images/compressed_dilated_edges.jpg')
print(f"Compressed image size: {compressed_size/1024:.2f} KB")