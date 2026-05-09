import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from ultralytics import YOLO
from recognition_logic import predict_formula_from_roi

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "shiki_structure_yolo_v8.pt")

try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"ERROR: YOLOモデルの読み込みに失敗しました。: {e}")

def merge_yolo_boxes(boxes, img_h, img_w):
    if not boxes: return []
    y_tol = img_h * 0.08
    x_tol = img_w * 0.15 

    def is_close(b1, b2):
        mid1 = (b1[1] + b1[3]) / 2
        mid2 = (b2[1] + b2[3]) / 2
        if abs(mid1 - mid2) > y_tol: return False
        dist_x = max(0, b2[0] - b1[2], b1[0] - b2[2])
        return dist_x < x_tol

    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        curr = boxes[i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if not used[j] and is_close(curr, boxes[j]):
                    curr = [
                        min(curr[0], boxes[j][0]),
                        min(curr[1], boxes[j][1]),
                        max(curr[2], boxes[j][2]),
                        max(curr[3], boxes[j][3])
                    ]
                    used[j] = True
                    changed = True
        merged.append(curr)
    return merged

@app.route('/', methods=['GET', 'POST'])
def index():
    equation_data = []
    main_image = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            main_image = 'uploads/' + file.filename
            
            img = cv2.imread(path)
            if img is None: return "画像が読み込めません。"
            h, w = img.shape[:2]

            results = yolo_model.predict(path, conf=0.2, verbose=True) 
            boxes = [b.xyxy[0].cpu().numpy().astype(int).tolist() for b in results[0].boxes if int(b.cls[0]) == 0]

            if boxes:
                final_boxes = merge_yolo_boxes(boxes, h, w)
                for i, (x1, y1, x2, y2) in enumerate(final_boxes):
                    box_h = y2 - y1
                    pad_h = max(20, int(box_h * 0.25))
                    pad_w = int(w * 0.05)
                    
                    rx1, ry1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                    rx2, ry2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
                    # ------------------------------------
                    
                    roi = img[ry1:ry2, rx1:rx2]
                    fname = f"line_{i}.png"
                    cv2.imwrite(os.path.join(UPLOAD_FOLDER, fname), roi)
                    
                    full_text, details = predict_formula_from_roi(roi)
                    
                    equation_data.append({
                        'img_url': 'uploads/' + fname,
                        'text': full_text,
                        'details': details,
                        'roi_size': [rx2 - rx1, ry2 - ry1]
                    })

    return render_template('index.html', equation_data=equation_data, main_image=main_image)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
