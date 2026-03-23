import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import cv2
import numpy as np
import torchvision.transforms as transforms

SYMBOL_MAP = {
    'alpha':'α', 'beta':'β', 'cong':'≅', 'divide':'÷', 'equal':'=', 
    'geq':'≥', 'greater':'>', 'infty':'∞', 'leq':'≤', 
    'less':'<', 'minus':'-', 'pi':'π', 'plus':'+', 
    'times':'×', 'var_a':'a', 'var_b':'b', 
    'var_c':'c', 'var_x':'x', 'var_y':'y', 'var_z':'z'
}
OP_CHARS = ['≅', '÷', '=', '≥', '>', '≤', '<', '-', '+', '×']

class MathFormulaCNN(nn.Module):
    def __init__(self, num_classes):
        super(MathFormulaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes) 
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

JSON_PATH = r"models\class_names.json"
SAVE_PATH = r"models\math_universal_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(JSON_PATH, 'r') as f:
    class_names = json.load(f)
model = MathFormulaCNN(len(class_names))
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.to(device).eval()

def resize_with_padding_white(img, size=(48, 48)):
    h, w = img.shape
    if h <= 0 or w <= 0: 
        return np.full(size, 255, dtype=np.uint8)
    margin = 12
    target_w, target_h = size[0] - margin, size[1] - margin
    
    scale = min(target_w / w, target_h / h)
    
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.full(size, 255, dtype=np.uint8)
    
    top = (size[1] - new_h) // 2
    left = (size[0] - new_w) // 2
    
    canvas[top:top+new_h, left:left+new_w] = resized
    
    return canvas


def merge_boxes(bboxes, threshold_dist=15):
    if not bboxes: return []
    bboxes = sorted(bboxes, key=lambda x: x[0])
    merged = []
    while len(bboxes) > 0:
        curr = list(bboxes.pop(0))
        found_merge = True
        while found_merge:
            found_merge = False
            for i in range(len(bboxes)):
                next_box = bboxes[i]
                overlap_x = max(0, min(curr[0]+curr[2], next_box[0]+next_box[2]) - max(curr[0], next_box[0]))
                dist_x = max(0, next_box[0] - (curr[0] + curr[2]), curr[0] - (next_box[0] + next_box[2]))
                dist_y = max(0, next_box[1] - (curr[1] + curr[3]), curr[1] - (next_box[1] + next_box[3]))
                is_vertical = overlap_x > min(curr[2], next_box[2]) * 0.5
                v_limit = 25 if is_vertical else threshold_dist
                if dist_x < 4 and dist_y < v_limit:
                    if not is_vertical and dist_y > 10: 
                        continue 

                    nx = min(curr[0], next_box[0])
                    ny = min(curr[1], next_box[1])
                    nw = max(curr[0]+curr[2], next_box[0]+next_box[2]) - nx
                    nh = max(curr[1]+curr[3], next_box[1]+next_box[3]) - ny
                    curr = [nx, ny, nw, nh]
                    bboxes.pop(i)
                    found_merge = True
                    break
        merged.append(tuple(curr))
    return merged

def predict_formula_from_roi(img_cv):
    if len(img_cv.shape) == 3: 
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    _, thresh_temp = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh_temp > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else (90 - angle if angle > 45 else -angle)
        M = cv2.getRotationMatrix2D((img_cv.shape[1]//2, img_cv.shape[0]//2), angle, 1.0)
        img_cv = cv2.warpAffine(img_cv, M, (img_cv.shape[1], img_cv.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    blurred = cv2.GaussianBlur(img_cv, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 30)
    
    cnt_img = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    initial_bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]
    temp_bboxes = merge_boxes(initial_bboxes, threshold_dist=15)
    temp_bboxes = sorted(temp_bboxes, key=lambda x: x[0])

    if not temp_bboxes: return "", []

    avg_width = np.mean([b[2] for b in temp_bboxes])
    final_bboxes = []
    for (x, y, w, h) in temp_bboxes:
        if w > avg_width * 1.8 and w > h * 1.3:
            roi_to_split = thresh[y:y+h, x:x+w]
            projection = np.sum(cv2.bitwise_not(roi_to_split), axis=0)
            s, e = int(w * 0.3), int(w * 0.7)
            split_rel = s + np.argmin(projection[s:e])
            final_bboxes.append((x, y, split_rel, h))
            final_bboxes.append((x + split_rel, y, w - split_rel, h))
        else:
            final_bboxes.append((x, y, w, h))

    res_text = []
    details = [] 
    prev_info = None 

    model.eval()
    for x, y, w, h in final_bboxes:
        roi_in = resize_with_padding_white(thresh[y:y+h, x:x+w], size=(48, 48))
        
        img_t = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])(roi_in).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(img_t)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1) 
            raw_symbol = class_names[pred.item()]
        
        char = SYMBOL_MAP.get(raw_symbol, raw_symbol.replace('var_',''))
        
        is_super = False
        if prev_info:
            px, py, pw, ph = prev_info
            if (y + h) < (py + ph * 0.6):
                is_super = True
        
        if is_super and res_text and res_text[-1] not in OP_CHARS:
            display_char = "^" + char
        else:
            display_char = char
            
        res_text.append(display_char)
        details.append({
            'char': display_char,
            'box': [x, y, w, h],
            'conf': float(conf.item()) 
        })
        prev_info = (x, y, w, h)
        
    return "".join(res_text), details