"""Test user images and log step-by-step timing for each image."""
import os
import time
import requests
import json
import cv2

BASE_URL = "http://127.0.0.1:5000"
IMAGES = [
    r"c:\Users\Abhinai\OneDrive\Desktop\Gemini\Art Restoration\sample_test.jpeg",
    r"c:\Users\Abhinai\OneDrive\Desktop\Gemini\Art Restoration\sample_test_2.jpeg"
]

print("=" * 70)
print("BENCHMARKING USER TEST IMAGES")
print("=" * 70)

for idx, img_path in enumerate(IMAGES, 1):
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        continue

    img = cv2.imread(img_path)
    h, w, c = img.shape
    size_kb = os.path.getsize(img_path) / 1024

    print(f"\n--- Testing Image #{idx}: {os.path.basename(img_path)} ---")
    print(f"Dimensions: {w}x{h} px | Size: {size_kb:.1f} KB")

    # 1. High Quality 4x mode
    print("\n  [Mode 4x - Ultra Quality]")
    start_time = time.time()
    with open(img_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/restore",
            files={"image": (os.path.basename(img_path), f, "image/jpeg")},
            data={"outscale": "4"}
        )
    total_time = time.time() - start_time

    print(f"  HTTP Status: {resp.status_code}")
    data = resp.json()
    if data.get("success"):
        details = data.get("details", {})
        print(f"  Output ID: {data['output_id']}")
        print(f"  Has Faces: {details.get('has_faces')}")
        print(f"  Device: {details.get('device')}")
        print(f"  Backend Processing Time: {details.get('processing_time')}s")
        print(f"  Total Roundtrip Time: {total_time:.2f}s")
    else:
        print(f"  Error: {data.get('error')}")

    # 2. Fast Mode 2x mode
    print("\n  [Mode 2x - Fast Mode]")
    start_time = time.time()
    with open(img_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/restore",
            files={"image": (os.path.basename(img_path), f, "image/jpeg")},
            data={"outscale": "2"}
        )
    total_time = time.time() - start_time

    print(f"  HTTP Status: {resp.status_code}")
    data = resp.json()
    if data.get("success"):
        details = data.get("details", {})
        print(f"  Output ID: {data['output_id']}")
        print(f"  Has Faces: {details.get('has_faces')}")
        print(f"  Device: {details.get('device')}")
        print(f"  Backend Processing Time: {details.get('processing_time')}s")
        print(f"  Total Roundtrip Time: {total_time:.2f}s")
    else:
        print(f"  Error: {data.get('error')}")

print("\n" + "=" * 70)
