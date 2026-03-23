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

# TODO: 
YOLO_MODEL_PATH = r"models\shiki_structure_yolo_v8.pt"

try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"ERROR: YOLOモデルの読み込みに失敗しました。パスを確認してください: {e}")

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

    merged_results = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        current_group = [boxes[i]]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if not used[j]:
                    for member in current_group:
                        if is_close(member, boxes[j]):
                            current_group.append(boxes[j])
                            used[j] = True
                            changed = True
                            break
        gx1 = min(b[0] for b in current_group)
        gy1 = min(b[1] for b in current_group)
        gx2 = max(b[2] for b in current_group)
        gy2 = max(b[3] for b in current_group)
        merged_results.append([gx1, gy1, gx2, gy2])
    return sorted(merged_results, key=lambda b: b[1])

@app.route('/', methods=['GET', 'POST'])
def index():
    main_image = None
    equation_data = []
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            path = os.path.join(UPLOAD_FOLDER, "input_main.png")
            file.save(path)
            main_image = 'uploads/input_main.png'
            
            img = cv2.imread(path)
            if img is None: return "画像が読み込めません。パスまたは形式を確認してください。"
            h, w = img.shape[:2]

            results = yolo_model.predict(path, conf=0.2, verbose=True) 
            boxes = [b.xyxy[0].cpu().numpy().astype(int).tolist() for b in results[0].boxes if int(b.cls[0]) == 0]

            if boxes:
                final_boxes = merge_yolo_boxes(boxes, h, w)
                for i, (x1, y1, x2, y2) in enumerate(final_boxes):
                    pad_h, pad_w = 15, int(w * 0.03)
                    rx1, ry1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                    rx2, ry2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
                    
                    roi = img[ry1:ry2, rx1:rx2]
                    fname = f"line_{i}.png"
                    cv2.imwrite(os.path.join(UPLOAD_FOLDER, fname), roi)
                    
                    full_text, details = predict_formula_from_roi(roi)
                    rh, rw = roi.shape[:2]
                    
                    equation_data.append({
                        'img_url': 'uploads/' + fname,
                        'text': full_text,
                        'details': details,
                        'roi_size': [rw, rh]
                    })
    
    return render_template('index.html', main_image=main_image, equation_data=equation_data)

@app.route('/terms')
def terms():
    return render_template('terms.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)