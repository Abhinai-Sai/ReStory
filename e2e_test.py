"""End-to-end test: upload image, restore, verify output, download."""
import requests
import os
import time
import json

BASE_URL = "http://127.0.0.1:5000"

print("=" * 60)
print("AI Art Restoration — End-to-End Test")
print("=" * 60)

# 1. Health check
print("\n[1] Health check...")
resp = requests.get(f"{BASE_URL}/api/health")
print(f"    Status: {resp.status_code}")
print(f"    Body: {json.dumps(resp.json(), indent=4)}")
assert resp.status_code == 200

# 2. Upload and restore
print("\n[2] Uploading test_image.jpg for restoration...")
with open("test_image.jpg", "rb") as f:
    start = time.time()
    resp = requests.post(f"{BASE_URL}/api/restore", files={"image": ("test_image.jpg", f, "image/jpeg")})
    elapsed = time.time() - start

print(f"    Status: {resp.status_code}")
data = resp.json()
print(f"    Response: {json.dumps(data, indent=4)}")
print(f"    Time: {elapsed:.1f}s")

if not data.get("success"):
    print(f"    ERROR: {data.get('error')}")
    exit(1)

output_id = data["output_id"]
print(f"    Output ID: {output_id}")

# 3. Fetch result
print(f"\n[3] Fetching result image /api/result/{output_id}...")
resp = requests.get(f"{BASE_URL}/api/result/{output_id}")
print(f"    Status: {resp.status_code}")
print(f"    Content-Type: {resp.headers.get('Content-Type')}")
print(f"    Size: {len(resp.content)} bytes")
assert resp.status_code == 200
assert len(resp.content) > 0

# 4. Download
print(f"\n[4] Downloading result /api/download/{output_id}...")
resp = requests.get(f"{BASE_URL}/api/download/{output_id}")
print(f"    Status: {resp.status_code}")
print(f"    Content-Disposition: {resp.headers.get('Content-Disposition')}")
print(f"    Size: {len(resp.content)} bytes")
assert resp.status_code == 200

# Save locally
with open("restored_test_output.jpg", "wb") as f:
    f.write(resp.content)
print(f"    Saved to: restored_test_output.jpg ({os.path.getsize('restored_test_output.jpg')} bytes)")

# 5. Invalid file test
print("\n[5] Testing invalid file rejection...")
resp = requests.post(f"{BASE_URL}/api/restore", files={"image": ("test.bmp", b"fake data", "image/bmp")})
print(f"    Status: {resp.status_code}")
print(f"    Error: {resp.json().get('error', 'none')}")
assert resp.status_code == 400

# 6. Missing file test
print("\n[6] Testing missing file rejection...")
resp = requests.post(f"{BASE_URL}/api/restore")
print(f"    Status: {resp.status_code}")
assert resp.status_code == 400

print("\n" + "=" * 60)
print("ALL END-TO-END TESTS PASSED!")
print("=" * 60)

