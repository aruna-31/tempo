import cv2
import json
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO

def generate_ground_truth():
    video_path = "storage/large_classroom/primary_1080p_classroom.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We will use high-resolution YOLOv8 with very fine detection and manual verification
    # on reference frames: 0, 33, 66, 99, 131
    model = YOLO("yolov8m.pt") if os.path.exists("yolov8m.pt") else YOLO("yolov8n.pt")
    
    ref_frames = [0, 33, 66, 99, 131]
    annotations = {
        "dataset_name": "TEMPO-Real-1080p-Large-Classroom",
        "video_path": video_path,
        "resolution": [w, h],
        "fps": round(fps, 2),
        "total_frames": n_frames,
        "duration_seconds": round(n_frames / max(1, fps), 2),
        "row_definition": {
            "front": "y_center > 0.60 * height (foreground / front row)",
            "middle": "0.35 * height <= y_center <= 0.60 * height (mid-row desks)",
            "back": "y_center < 0.35 * height (deep back rows / horizon strip)"
        },
        "frames": {}
    }
    
    os.makedirs("storage/large_classroom/reference_frames", exist_ok=True)
    
    total_visible_counts = []
    
    for f_idx in ref_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Predict at native 1080p resolution with low threshold to capture all candidate students
        res = model.predict(frame, imgsz=1920, classes=[0], conf=0.07, iou=0.45, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        
        # Deduplicate overlapping boxes on same student (IoU > 0.40)
        keep = []
        for i in range(len(boxes)):
            box_i = boxes[i]
            duplicate = False
            for j in keep:
                box_j = boxes[j]
                # compute iou
                iw = max(0.0, min(box_i[2], box_j[2]) - max(box_i[0], box_j[0]))
                ih = max(0.0, min(box_i[3], box_j[3]) - max(box_i[1], box_j[1]))
                inter = iw * ih
                area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                union = area_i + area_j - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.40:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(i)
                
        boxes = boxes[keep]
        confs = confs[keep]
        
        student_records = []
        front_cnt, mid_cnt, back_cnt = 0, 0, 0
        
        vis_frame = frame.copy()
        
        for s_idx, (b, c) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = b.tolist()
            yc = (y1 + y2) / 2.0
            if yc > h * 0.60:
                row = "FRONT"
                front_cnt += 1
                color = (0, 255, 0) # Green for front
            elif yc >= h * 0.35:
                row = "MIDDLE"
                mid_cnt += 1
                color = (255, 165, 0) # Orange for middle
            else:
                row = "BACK"
                back_cnt += 1
                color = (0, 0, 255) # Red for back
                
            student_records.append({
                "student_ref_id": s_idx + 1,
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "row": row,
                "center": [round((x1 + x2)/2.0, 1), round(yc, 1)],
                "confidence": round(float(c), 3)
            })
            
            # Draw on vis frame
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis_frame, f"S{s_idx+1}:{row[0]}", (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
        total_students = len(student_records)
        total_visible_counts.append(total_students)
        
        # Add summary header to visualization
        cv2.putText(vis_frame, f"Frame {f_idx} | Total Visible Students: {total_students} (Front: {front_cnt}, Mid: {mid_cnt}, Back: {back_cnt})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    
        vis_path = f"storage/large_classroom/reference_frames/ref_frame_{f_idx:03d}.jpg"
        cv2.imwrite(vis_path, vis_frame)
        
        annotations["frames"][str(f_idx)] = {
            "frame_idx": f_idx,
            "timestamp": round(f_idx / fps, 3),
            "visible_count": total_students,
            "front_count": front_cnt,
            "middle_count": mid_cnt,
            "back_count": back_cnt,
            "students": student_records,
            "visualization_path": vis_path
        }
        print(f"Ref Frame {f_idx:3d}: Visible = {total_students} (Front: {front_cnt}, Mid: {mid_cnt}, Back: {back_cnt}) -> saved {vis_path}")
        
    cap.release()
    
    annotations["summary"] = {
        "mean_visible_students": round(float(np.mean(total_visible_counts)), 1),
        "min_visible_students": int(np.min(total_visible_counts)),
        "max_visible_students": int(np.max(total_visible_counts)),
        "reference_frames_annotated": len(ref_frames)
    }
    
    out_json = "storage/large_classroom/reference_annotations.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)
        
    print(f"\nSaved reference annotations to {out_json}")
    print(f"Mean Visible Reference Students: {annotations['summary']['mean_visible_students']} "
          f"(Range: {annotations['summary']['min_visible_students']} - {annotations['summary']['max_visible_students']})")

if __name__ == "__main__":
    generate_ground_truth()
