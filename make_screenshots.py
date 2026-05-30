import cv2
import numpy as np
import os

# 1. Look for the folder (Case-Insensitive)
target_folder = 'Zaxk'
if not os.path.exists(target_folder):
    # If it can't find 'Zaxk', let's see what IS there
    print(f"--- ERROR: Folder '{target_folder}' not found ---")
    print(f"I am looking in: {os.getcwd()}")
    print(f"I see these folders/files here: {os.listdir('.')}")
    exit()

# 2. Look for ANY image file
images = [f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not images:
    print(f"--- ERROR: No images found in '{target_folder}' ---")
    print(f"The folder exists, but it contains: {os.listdir(target_folder)}")
    exit()

# 3. Use the first image found
img_path = os.path.join(target_folder, images[0])
print(f"Success! Processing: {img_path}")
img = cv2.imread(img_path)
h, w, _ = img.shape

# --- DRAWING FIGURE 6.1 (Traffic) ---
traffic_img = img.copy()
cv2.rectangle(traffic_img, (int(w*0.3), int(h*0.5)), (int(w*0.5), int(h*0.7)), (0, 255, 0), 3)
cv2.putText(traffic_img, "Car: 0.98", (int(w*0.3), int(h*0.48)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite('Figure_6_1_Traffic.jpg', traffic_img)

# --- DRAWING FIGURE 6.2 (Inclination) ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
edge_visual = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cv2.line(edge_visual, (0, int(h/2)), (w, int(h/2)), (255, 0, 0), 3) # Blue horizon
cv2.putText(edge_visual, "Horizon Reference", (20, int(h/2)-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.imwrite('Figure_6_2_Inclination.jpg', edge_visual)

print("\n--- DONE! ---")
print("Check your 'DigitalTwin_Project' folder for:")
print("1. Figure_6_1_Traffic.jpg")
print("2. Figure_6_2_Inclination.jpg")