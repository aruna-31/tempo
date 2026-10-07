import cv2
import os
import json
from pathlib import Path
from ultralytics import YOLO

def analyze_video(video_path: str, name: str):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total_frames / max(1, fps)
    
    print(f"\n=======================================================")
    print(f"  Analyzing: {name}")
    print(f"  Path: {video_path}")
    print(f"  Resolution: {w}x{h} | FPS: {fps:.2f} | Frames: {total_frames} | Duration: {dur:.2f}s")
    print(f"=======================================================")
    
    # Load model
    model = YOLO("yolov8n.pt")
    
    # Grab middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame.")
        return
        
    for conf_thresh in [0.05, 0.10, 0.15, 0.25]:
        res = model(frame, imgsz=1920 if w >= 1920 else 1280, classes=[0], conf=conf_thresh, verbose=False)
        boxes = res[0].boxes.xyxy.cpu().numpy()
        confs = res[0].boxes.conf.cpu().numpy()
        
        front, mid, back = [], [], []
        for b, c in zip(boxes, confs):
            x1, y1, x2, y2 = b
            yc = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            # Front: bottom 40% (yc > 0.60)
            # Middle: 30% to 60% (0.30 <= yc <= 0.60)
            # Back: top 30% (yc < 0.30)
            if yc > h * 0.58:
                front.append((b, c, area))
            elif yc > h * 0.32:
                mid.append((b, c, area))
            else:
                back.append((b, c, area))
                
        print(f"  Conf >= {conf_thresh:.2f}: Total Detected = {len(boxes):2d} | Front: {len(front):2d} | Mid: {len(mid):2d} | Back: {len(back):2d}")

if __name__ == "__main__":
    analyze_video(r"c:\Users\aruna\Downloads\VID20260922112800~2.mp4", "1080p Real Large Classroom")
    analyze_video(r"c:\Users\aruna\Downloads\classroom.mp4", "Classroom 848x478")
