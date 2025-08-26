import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from sort import Sort  # Make sure sort.py is in the same folder or installed

class VLIS_Detector:
    def __init__(self, model_path='license_plate_detector.pt'):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'])
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.results_dir = 'results'
        self.logs_dir = 'logs'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def _ocr_plate(self, plate_img):
        try:
            result = self.reader.readtext(plate_img)
            texts = [res[1] for res in result if len(res[1]) >= 4]  # filter very short texts
            if texts:
                return max(texts, key=len)
            else:
                return ""
        except Exception:
            return ""

    def detect_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image could not be loaded.")

        results = self.model(img)[0]
        plates = []
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cropped_plate = img[y1:y2, x1:x2]
            text = self._ocr_plate(cropped_plate)
            plates.append((text, (x1, y1, x2, y2), cropped_plate))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            crop_filename = os.path.join(self.results_dir, f"img_plate_{timestamp}.jpg")
            cv2.imwrite(crop_filename, cropped_plate)

            log_file = os.path.join(self.logs_dir, "image_detections.txt")
            with open(log_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")

        return img, plates

class VLIS_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VLIS - Vehicle License Identification System")
        self.root.geometry("1100x700")
        self.root.configure(bg="#121212")

        self.detector = VLIS_Detector()

        # Left vertical menu frame
        menu_frame = tk.Frame(root, bg="#1F1F1F", width=180)
        menu_frame.pack(side='left', fill='y')

        btn_params = {'font': ('Segoe UI', 12), 'bg': '#292929', 'fg': '#E0E0E0',
                      'activebackground': '#3A3A3A', 'activeforeground': '#FFFFFF', 'bd': 0, 'relief': 'flat', 'padx': 10, 'pady': 10}

        tk.Label(menu_frame, text="VLIS Menu", font=('Segoe UI', 16, 'bold'),
                 bg="#1F1F1F", fg="#00FF00").pack(pady=15)

        tk.Button(menu_frame, text="Load Image", command=self.load_image, **btn_params).pack(fill='x', pady=10, padx=10)
        tk.Button(menu_frame, text="Load Video", command=self.load_video, **btn_params).pack(fill='x', pady=10, padx=10)
        tk.Button(menu_frame, text="Clear Screen", command=self.clear_screen, **btn_params).pack(fill='x', pady=10, padx=10)
        tk.Button(menu_frame, text="Exit", command=self.exit_app, **btn_params).pack(fill='x', pady=10, padx=10)

        # Main display canvas
        self.display_canvas = tk.Canvas(root, bg="#222222", highlightthickness=0)
        self.display_canvas.pack(side='right', expand=True, fill='both')

        # Bottom text preview for plates
        self.plate_preview = tk.Text(root, height=6, bg="#121212", fg="#00FF00",
                                     font=('Consolas', 14), bd=0, relief='flat')
        self.plate_preview.pack(side='bottom', fill='x', padx=10, pady=(0, 10))

        self.current_image = None
        self.video_thread = None
        self.stop_video_flag = False

    def load_image(self):
        self.clear_screen()
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        img_path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        if not img_path:
            return
        try:
            img, plates = self.detector.detect_image(img_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{e}")
            return

        self.show_image(img)
        self.update_plate_preview(plates)

    def load_video(self):
        self.clear_screen()
        if self.video_thread and self.video_thread.is_alive():
            messagebox.showwarning("Video Running", "Video is already running.")
            return
        filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        vid_path = filedialog.askopenfilename(title="Select Video", filetypes=filetypes)
        if not vid_path:
            return

        self.stop_video_flag = False
        self.video_thread = threading.Thread(target=self.process_video, args=(vid_path,), daemon=True)
        self.video_thread.start()

    def process_video(self, vid_path):
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            return

        plate_logs = {}  # track_id -> list of detected plates
        last_crops = {}  # track_id -> last cropped plate image

        while True:
            if self.stop_video_flag:
                break
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detector.model(frame)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            dets_for_sort = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                if len(box) > 4:
                    conf = box[4]
                else:
                    conf = 0.5  # default confidence if missing
                dets_for_sort.append([x1, y1, x2, y2, conf])
            dets_for_sort = np.array(dets_for_sort) if dets_for_sort else np.empty((0, 5))

            tracked_objects = self.detector.tracker.update(dets_for_sort)
            plates_this_frame = []

            for *box, track_id in tracked_objects:
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                # Validate crop coords
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue  # skip invalid boxes

                cropped_plate = frame[y1:y2, x1:x2]
                text = self.detector._ocr_plate(cropped_plate).strip()

                if track_id not in plate_logs:
                    plate_logs[track_id] = []
                if text:
                    plate_logs[track_id].append(text)
                    last_crops[track_id] = cropped_plate.copy()

                plates_this_frame.append((int(track_id), text))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {int(track_id)}: {text}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.show_image(frame)
            self.update_plate_preview(plates_this_frame)
            # Removed cv2.waitKey() per your environment

        cap.release()

        # Save the most frequent plate per track_id
        # Save the most frequent plate per track_id
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.detector.logs_dir,
                                f"video_{os.path.splitext(os.path.basename(vid_path))[0]}_{timestamp}.txt")

        with open(log_file, 'a') as f:
            for tid in plate_logs:
                texts = plate_logs[tid]
                if not texts:
                    continue
                most_freq_plate = max(set(texts), key=texts.count)
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Track ID {tid}: {most_freq_plate}\n")

                # Save cropped plate image
                crop_img = last_crops.get(tid)
                if crop_img is not None:
                    crop_filename = os.path.join(self.detector.results_dir, f"video_plate_id{tid}_{timestamp}.jpg")
                    cv2.imwrite(crop_filename, crop_img)

        messagebox.showinfo("Video Processing", "Video processing completed! Results saved.")

    def show_image(self, cv_img):
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img_rgb)
        w, h = pil_img.size
        max_w, max_h = self.display_canvas.winfo_width(), self.display_canvas.winfo_height()
        if max_w == 1 or max_h == 1:  # initial size may be 1, so set defaults
            max_w, max_h = 900, 600
        scale = min(max_w/w, max_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.current_image = ImageTk.PhotoImage(pil_img)
        self.display_canvas.delete("all")
        self.display_canvas.create_image(max_w//2, max_h//2, image=self.current_image, anchor='center')

    def update_plate_preview(self, plates):
        self.plate_preview.delete(1.0, tk.END)
        if not plates:
            self.plate_preview.insert(tk.END, "No plates detected.")
            return
        if isinstance(plates[0], tuple):
            if len(plates[0]) == 3:  # image detection output
                for text, bbox, _ in plates:
                    self.plate_preview.insert(tk.END, f"{text}\n")
            else:  # video detection (id, text)
                for tid, text in plates:
                    self.plate_preview.insert(tk.END, f"ID {tid}: {text}\n")

    def clear_screen(self):
        self.display_canvas.delete("all")
        self.plate_preview.delete(1.0, tk.END)

    def exit_app(self):
        self.stop_video_flag = True
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VLIS_GUI(root)
    root.mainloop()
