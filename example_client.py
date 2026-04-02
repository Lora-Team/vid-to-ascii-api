# ============================================
# How to request your vid-to-ascii API
# ============================================

import requests
import json

API_URL = "https://vid-to-ascii-api.vercel.app/"

# --- POST request to convert a video ---
response = requests.post(API_URL, json={
    "url": "https://example.com/somevideo.mp4",  # The video URL
    "fps": 10,          # Optional: frames per second to extract (default: 10)
    "max_frames": 100   # Optional: max frames to return (default: 100)
})

data = response.json()

# --- What you get back ---
# {
#   "frames": [
#       "<#FF5733>⏹<#000000>⏹<#FFFFFF>⏹...\n<#AA00BB>⏹...",   <-- frame 1
#       "<#112233>⏹<#445566>⏹...\n...",                          <-- frame 2
#       ...
#   ],
#   "frame_count": 42,
#   "dimensions": "80x20",
#   "black_and_white": false,
#   "fps": 10
# }

print(f"Got {data['frame_count']} frames")
print(f"Dimensions: {data['dimensions']}")
print(f"Black & White: {data['black_and_white']}")

# Print first frame
if data["frames"]:
    print("\n--- Frame 1 ---")
    print(data["frames"][0])
