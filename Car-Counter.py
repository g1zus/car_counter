import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import torch

video_capture = cv2.VideoCapture("dram_m_2.mp4")

codec = cv2.VideoWriter_fourcc(*'mp4v')
video_output = cv2.VideoWriter('output_video.mp4', codec, 30.0, (int(video_capture.get(3)), int(video_capture.get(4))))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = YOLO("../Yolo-Weights/yolov9c.pt").to(device)

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
          "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
          "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"]

mask_image = cv2.imread("mask_m.png")

object_tracker = Sort(max_age=15, min_hits=4, iou_threshold=0.3)

lines = [
    [144, 315, 252, 301],
    [428, 454, 539, 418],
    [128, 352, 151, 457],
    [332, 560, 465, 500]
]

count_tank = []
count_books = []
count_dram = []
count_loft = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    masked_frame = cv2.bitwise_and(frame, mask_image)
    detections_stream = model(masked_frame, stream=True)

    all_detections = np.empty((0, 5))

    for detection in detections_stream:
        boxes = detection.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width, height = x2 - x1, y2 - y1
            confidence = round(box.conf[0].item(), 2)
            class_id = int(box.cls[0])
            label = labels[class_id]
            if label in ["car", "truck", "bus", "motorbike"] and confidence > 0.3:
                detection_array = np.array([x1, y1, x2, y2, confidence])
                all_detections = np.vstack((all_detections, detection_array))

    tracked_objects = object_tracker.update(all_detections)

    for line in lines:
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        width, height = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, width, height), l=9, rt=1, t=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f' {obj_id}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1, offset=1)

        center_x, center_y = x1 + width // 2, y1 + height // 2
        print(f'id: {obj_id} cx: {center_x}, cy: {center_y}')
        cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), cv2.FILLED)

        if lines[0][0] < center_x < lines[0][2] and lines[0][1] - 25 < center_y < lines[0][1] and obj_id not in count_tank:
            count_tank.append(obj_id)
            cv2.line(frame, (lines[0][0], lines[0][1]), (lines[0][2], lines[0][3]), (0, 255, 0), 2)
        if lines[1][0] < center_x < lines[1][2] and lines[1][3] - 10 < center_y < lines[1][1] and obj_id not in count_dram:
            count_dram.append(obj_id)
            cv2.line(frame, (lines[1][0], lines[1][1]), (lines[1][2], lines[1][3]), (0, 255, 0), 2)
        if lines[2][0] < center_x < lines[2][2] and lines[2][1] < center_y < lines[2][3] and obj_id not in count_books:
            count_books.append(obj_id)
            cv2.line(frame, (lines[2][0], lines[2][1]), (lines[2][2], lines[2][3]), (0, 255, 0), 2)
        if lines[3][0] < center_x < lines[3][2] and lines[3][3] < center_y < lines[3][1] and obj_id not in count_loft:
            count_loft.append(obj_id)
            cv2.line(frame, (lines[3][0], lines[3][1]), (lines[3][2], lines[3][3]), (0, 255, 0), 2)

    print(f'Tank: {count_tank}', f'Books: {count_books}', f'Dram: {count_dram}', f'Loft: {count_loft}', sep='\n')

    cvzone.putTextRect(frame, f'Tank: {len(count_tank)}', (190, 585), 0.8, 1, (0, 0, 0), (255, 255, 255), cv2.FONT_ITALIC)
    cvzone.putTextRect(frame, f'Dram: {len(count_dram)}', (190, 615), 0.8, 1, (0, 0, 0), (255, 255, 255), cv2.FONT_ITALIC)
    cvzone.putTextRect(frame, f'Books: {len(count_books)}', (190, 645), 0.8, 1, (0, 0, 0), (255, 255, 255), cv2.FONT_ITALIC)
    cvzone.putTextRect(frame, f'Loft: {len(count_loft)}', (190, 675), 0.8, 1, (0, 0, 0), (255, 255, 255), cv2.FONT_ITALIC)

    cv2.imshow("God's Eye", frame)
    video_output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
video_output.release()
cv2.destroyAllWindows()