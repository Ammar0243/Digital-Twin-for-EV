import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import pandas as pd
import numpy as np

# 1. Load the AI for counting traffic
print("Waking up the AI model...")
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    data = []
    frame_idx = 0
    
    print("Processing video... this may take a moment.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Process 1 frame every second (assuming 30fps video)
        if frame_idx % 30 == 0:
            # --- FEATURE 1: TRAFFIC DENSITY ---
            # Turn frame into a format the AI understands
            img = torchvision.transforms.functional.to_tensor(frame)
            with torch.no_grad():
                pred = model([img])
            
            # Count boxes that the AI is 50% sure are cars (label 3), buses (6), or trucks (8)
            traffic_count = sum(1 for label, score in zip(pred[0]['labels'], pred[0]['scores']) 
                                if label in [3, 6, 8] and score > 0.5)

            # --- FEATURE 2: INCLINATION ---
            # We look for the 'horizon' line. If it shifts down, the road is going UP.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            
            incline_val = 0.0
            if lines is not None:
                # Calculate average height of detected lines
                avg_y = np.mean([line[0][1] for line in lines])
                # Convert pixel shift to a simple "Incline Score"
                incline_val = round((avg_y - (frame.shape[0] / 2)) / 50, 2)

            data.append({
                "time_minute": frame_idx // (30 * 60), # Syncs with groupmate's CSV
                "traffic_density": traffic_count,
                "inclination": incline_val
            })
            print(f"Processed Second {frame_idx//30} | Traffic: {traffic_count} | Incline: {incline_val}")

        frame_idx += 1
    
    cap.release()
    return pd.DataFrame(data)

# Run the processor
video_features = process_video('dashcam.mp4')
video_features.to_csv('extracted_features.csv', index=False)
print("\nDone! Check your folder for 'extracted_features.csv'")