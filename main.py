import cv2
from ultralytics import YOLO
import threading
import queue
import numpy as np

def load_model(model_path):
    model = YOLO(model_path)
    return model

def detect_objects(model, image):
    results = model(image)
    return results[0].boxes.data.cpu().numpy()

def process_frame(model, frame):
    detections = detect_objects(model, frame)
    boxes = []
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        if confidence > 0.5:  # Adjust confidence threshold as needed
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            class_name = model.names[int(class_id)]
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            boxes.append((x1, y1, x2, y2, class_id))
    return frame, boxes

def calculate_distance(box1, box2, focal_length, baseline):
    disparity = abs(box1[0] - box2[0])  # Assuming box1 and box2 are (x1, y1, x2, y2, class_id)
    if disparity == 0:
        return float('inf')  # Avoid division by zero
    distance = (focal_length * baseline) / disparity
    return distance

def video_stream(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

def main():
    model_path = 'yolov5su.pt'  # or use 'yolov5n' for a pre-trained model
    model = load_model(model_path)

    cam_index1 = 0
    cam_index2 = 2

    cap1 = cv2.VideoCapture(cam_index1)
    cap2 = cv2.VideoCapture(cam_index2)

    # cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frame_queue1 = queue.Queue(maxsize=4)
    frame_queue2 = queue.Queue(maxsize=4)

    # Stereo vision parameters
    focal_length = 400  # Adjust based on your camera
    baseline = 0.12  # Adjust based on your setup (distance between cameras in meters)

    # Start video streaming threads
    thread1 = threading.Thread(target=video_stream, args=(cap1, frame_queue1))
    thread2 = threading.Thread(target=video_stream, args=(cap2, frame_queue2))
    thread1.start()
    thread2.start()

    while True:
        if frame_queue1.empty() or frame_queue2.empty():
            continue

        frame1 = frame_queue1.get()
        frame2 = frame_queue2.get()

        # Process both frames
        processed_frame1, boxes1 = process_frame(model, frame1)
        processed_frame2, boxes2 = process_frame(model, frame2)

        # Resize frames to fit side by side
        processed_frame1 = cv2.resize(processed_frame1, (640, 480))
        processed_frame2 = cv2.resize(processed_frame2, (640, 480))

        # Match objects and calculate distance
        for box1 in boxes1:
            for box2 in boxes2:
                if box1[4] == box2[4]:  # If class IDs match
                    distance = calculate_distance(box1, box2, focal_length, baseline)
                    label = f'Distance: {distance:.2f}m'
                    cv2.putText(processed_frame1, label, (box1[0], box1[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.putText(processed_frame2, label, (box2[0], box2[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Concatenate frames horizontally
        combined_frame = cv2.hconcat([processed_frame1, processed_frame2])
        
        print_txt = f"cam index = {cam_index1} and second = {cam_index2}"
        cv2.imshow(print_txt, combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    thread1.join()
    thread2.join()

if __name__ == '__main__':
    main()
