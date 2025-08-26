# detect.py
import os
import cv2
import torch
import easyocr
import numpy as np
from tracker import Sort
from datetime import datetime

class VLISDetector:
    def __init__(self, model_path='license_plate_detector.pt', device='cpu'):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path, device=device)
        self.reader = easyocr.Reader(['en'], gpu=(device!='cpu'))
        self.tracker = Sort()
        self.results_dir = 'results'
        self.logs_dir = 'logs'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def detect_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image not found or unable to read")

        detections = self.model(img)

        boxes = detections.xyxy[0].cpu().numpy()  # xyxy + conf + class
        plates = []

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            plate_img = img[y1:y2, x1:x2]
            text = self._ocr_plate(plate_img)
            plates.append((text, (x1, y1, x2, y2), plate_img))

            # Draw bounding box + text
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

            # Save cropped plate
            self._save_plate_and_log(text, plate_img, img_path, img)

        return img, plates

    def detect_video(self, video_path, gui_callback=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video")

        plate_logs = {}
        frame_count = 0
        ret = True
        while ret:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            detections = self.model(frame)
            boxes = detections.xyxy[0].cpu().numpy()

            dets_for_sort = []
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                dets_for_sort.append([x1, y1, x2, y2, conf])
            dets_for_sort = np.array(dets_for_sort)

            tracked_objects = self.tracker.update(dets_for_sort)

            for *box, track_id in tracked_objects:
                x1, y1, x2, y2 = map(int, box)
                cropped_plate = frame[y1:y2, x1:x2]
                text = self._ocr_plate(cropped_plate)
                if track_id not in plate_logs:
                    plate_logs[track_id] = []
                if text.strip():
                    plate_logs[track_id].append(text.strip())

                # Draw bounding box and text on frame
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f'ID {int(track_id)}: {text}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if gui_callback:
                gui_callback(frame)

        cap.release()

        # After video ends, save most frequent plates per vehicle
        results = []
        for track_id, texts in plate_logs.items():
            if len(texts) == 0:
                continue
            # Most frequent plate text
            plate_text = max(set(texts), key=texts.count)
            results.append((track_id, plate_text))

        # Save results & logs
        for track_id, plate_text in results:
            # Get last frame with that plate
            # For simplicity, we skip re-extracting cropped plates here
            log_file = os.path.join(self.logs_dir, f'video_{os.path.basename(video_path)}.txt')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_line = f"[{timestamp}] Track ID {track_id}: {plate_text}\n"
            with open(log_file, 'a') as f:
                f.write(log_line)

        return results

    def _ocr_plate(self, plate_img):
        if plate_img.size == 0:
            return ""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # Optional thresholding can be done here
        result = self.reader.readtext(gray)
        text = ""
        if result:
            text = ' '.join([res[1] for res in result]).strip()
        return text

    def _save_plate_and_log(self, text, plate_img, src_path, full_img):
        if text.strip() == "":
            return
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cropped_plate_path = os.path.join(self.results_dir, f"{base_name}_plate_{timestamp}.jpg")
        highlighted_img_path = os.path.join(self.results_dir, f"{base_name}_highlighted_{timestamp}.jpg")

        # Save cropped plate image
        cv2.imwrite(cropped_plate_path, plate_img)
        # Save highlighted full image with bounding box and text
        cv2.imwrite(highlighted_img_path, full_img)

        # Save log text
        log_file = os.path.join(self.logs_dir, f"{base_name}.txt")
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(f"[{time_str}] Plate: {text}\n")
