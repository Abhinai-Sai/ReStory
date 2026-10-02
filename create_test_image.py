"""Create a test image for end-to-end testing."""
import cv2
import numpy as np
import os

# Create a simple test image (simulating a damaged/degraded photo)
img = np.zeros((200, 300, 3), dtype=np.uint8)
cv2.rectangle(img, (20, 20), (280, 180), (80, 120, 180), -1)
cv2.rectangle(img, (50, 50), (250, 150), (40, 60, 100), -1)
cv2.circle(img, (150, 100), 40, (200, 180, 140), -1)

# Add noise to simulate damage
noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
img = cv2.add(img, noise)

cv2.imwrite("test_image.jpg", img)
size = os.path.getsize("test_image.jpg")
print(f"Test image created: {size} bytes, shape: {img.shape}")

