import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from ultralytics import YOLO
from bisect import bisect_left

from logic_system import predict_formula_main

app = Flask(__name__)
base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, 'static', 'uploads')
os.makedirs(save_dir, exist_ok=True)
yolo_path = os.path.join(base_dir, "models", "shiki_structure_yolo_v8.pt")

try:
    yolo_model = YOLO(yolo_path)
except Exception as e:
    print(f"ERROR: YOLO model load failed: {e}")

def resize(img, tw=1900, th=2700):
    """画像を指定サイズ（1900x2700）に収まるよう正規化する"""
    # img.shape[:2] 画像の高さ（h）と幅（w）を取得。  
    # cv2.resize() 画像が指定サイズより大きい場合、領域補間法を用いて縮小。  
    # np.full((th, tw, 3), 255, dtype=np.uint8) 指定サイズの白い背景を作成。  
    h, w = img.shape[:2]
    
    if w <= tw and h <= th:
        return img
        
    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
    dx = (tw - new_w) // 2
    dy = (th - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized# キャンバスへの配置: 縮小した画像を計算した座標に基づき、中央に配置します。  
    
    return canvas

def remove(img, val=150):
    """画像からノイズを除去し二値化してクリーンな画像を作成"""
    # cv2.cvtColor() カラー画像をグレースケールに変換。  
    # cv2.GaussianBlur() 画像をぼかして微細なノイズを低減。  
    # cv2.threshold() 指定した値で白黒に二値化し、計算式のみを抽出。  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, val, 255, cv2.THRESH_BINARY)
    clean_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return clean_img

def select(boxes, iou=0.5, ratio=0.7):
    """YOLOで検出された重複するボックスを整理するアルゴリズム"""
    # arr = sorted() ボックスを面積の大きい順に並べ替。  
    # inter / union (IoU): 2つのボックスの重なり具合を計算します。  
    # drop.add(other) 重なりが閾値（iouまたはratio）を超えた場合、小さい方のボックスを削除対象とする。  
    if not boxes:
        return []

    arr = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        area = (x2 - x1) * (y2 - y1)
        arr.append({
            'box': [x1, y1, x2, y2],
            'area': area,
            'idx': idx
        })
    
    arr = sorted(arr, key=lambda x: x['area'], reverse=True)
    
    n = len(boxes)
    keep = set(range(n))
    drop = set()

    for i in range(n):
        cur = arr[i]['idx']
        if cur in drop:
            continue
            
        ax1, ay1, ax2, ay2 = arr[i]['box']
        area_a = arr[i]['area']

        for j in range(i + 1, n):
            other = arr[j]['idx']
            if other in drop:
                continue

            bx1, by1, bx2, by2 = arr[j]['box']
            area_b = arr[j]['area']

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih

            if inter > 0:
                union = area_a + area_b - inter
                cur_iou = inter / union
                cur_ratio = inter / area_b

                if cur_iou > iou or cur_ratio > ratio:
                    drop.add(other)
                    if other in keep:
                        keep.remove(other)

    return [boxes[k] for k in range(n) if k in keep]


@app.route('/', methods=['GET', 'POST'])
def index():
    """画像アップロードから数式抽出までの一連の流れを制御"""
    # file.save(temp_path) 送信された画像を保存する。  
    # yolo_model.predict(..., conf=0.5) YOLOを用いて画像から数式の領域を検出する。  
    # bisect_left() 検出されたボックスを、y座標に基づいて上から順にソート。  
    # 検出した領域にパディングを加えて切り出し、predict_formula_main を呼び出してLaTeXへ変換します。
    equation_data = []
    main_img = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            temp_path = os.path.join(save_dir, "temp_orig_" + file.filename)
            file.save(temp_path)
            
            img = cv2.imread(temp_path)
            if img is None: 
                return "画像が読み込めません。"
            
            res_img = resize(img, tw=1900, th=2700)
            
            path = os.path.join(save_dir, file.filename)
            cv2.imwrite(path, res_img)
            main_img = 'uploads/' + file.filename
            
            h, w = res_img.shape[:2]

            clean_img = remove(res_img, val=180)
            
            clean_path = os.path.join(save_dir, "clean_temp_" + file.filename)
            cv2.imwrite(clean_path, clean_img)

    
            results = yolo_model.predict(clean_path, conf=0.5, verbose=True)
            raw_box = [b.xyxy[0].cpu().numpy().astype(int).tolist() for b in results[0].boxes if int(b.cls[0]) == 0]

            sel_box = select(raw_box, iou=0.5, ratio=0.7)

            if sel_box:
                sort_box = []
                for b in sel_box:
                    ys = [box[1] for box in sort_box]
                    idx = bisect_left(ys, b[1])
                    sort_box.insert(idx, b)
                
                box_num = len(sort_box)
                for i in range(1, box_num + 1):
                    x1, y1, x2, y2 = sort_box[i - 1]
                    
                    box_h = y2 - y1
                    pad_h = max(20, int(box_h * 0.25))
                    pad_w = int(w * 0.05)
                    
                    rx1, ry1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                    rx2, ry2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
                    
                    roi = clean_img[ry1:ry2, rx1:rx2]
                    fname = f"line_{i}.png"
                    cv2.imwrite(os.path.join(save_dir, fname), roi)
                    
                    full_text, details = predict_formula_main(roi)
                    
                    equation_data.append({
                        'img_url': 'uploads/' + fname,
                        'text': full_text,
                        'details': details,
                        'roi_size': [rx2 - rx1, ry2 - ry1]
                    })
            
            for t_path in [temp_path, clean_path]:
                if os.path.exists(t_path):
                    os.remove(t_path)

    return render_template('index.html', equation_data=equation_data, main_image=main_img)

@app.route('/terms')
def terms():
    return render_template('terms.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)