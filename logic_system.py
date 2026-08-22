import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import cv2
import numpy as np
import torchvision.transforms as transforms

symbolmap = {
    'alpha': '\\alpha ', 'beta': '\\beta ', 'round_d': '\\partial ', 'divide': '\\div ', 'equal': '=', 
    'greater': '>', 'infty': '\\infty ', 'r_sp': ')', 'omega': '\\omega ', 'phi': '\\phi ', 'epsilon': '\\epsilon ',  
    'less': '<', 'minus': '-', 'pi': '\\pi ', 'plus': '+', 'times': '\\times ', 'delta': '\\Delta ',
    'sigma': '\\sum ', 'theta': '\\theta ', 'integral': '\\int ', 'frac_line': '\\frac',
    'var_n': 'n', 'var_a': 'a', 'var_b': 'b', 'var_c': 'c', 'var_d': 'd',
    'var_s': 's', 'var_x': 'x', 'var_y': 'y', 'var_z': 'z', 'var_e': 'e',
    'var_p': 'p', 'var_k': 'k', 'var_t': 't', 'var_v': 'v',
    'war_V': 'V', 'war_W': 'W', 'war_T': 'T', 'war_N': 'N', 'war_K': 'K'
}

OP_CHARS = ['\\div ', '=', '>', '<', '-', '+', '\\times ', '\\sum ', '\\int ', '\\frac']

basepath = os.path.dirname(os.path.abspath(__file__))
jsonpath = os.path.join(basepath, "models", "class_names.json")
modelpath = os.path.join(basepath, "models", "math_universal_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MathFormulaCNN(nn.Module):
    """数式文字を認識するためのニューラルネットワーク構造とその初期化設定"""
    # __init__: 3層の畳み込み層で特徴を抽出し、2層の全結合層で最終的な文字クラスに分類するCNN構造を定義。dropoutで過学習を抑制する。
    # forward: 入力画像に対し、ReLU活性化関数とプーリング層を適用して特徴を圧縮し、最終的にクラスごとのスコアを出力する。
    # モデル読み込み: JSONから文字リストを取得し、学習済み重みをロードして評価モードに設定。
    def __init__(self, num_classes):
        super(MathFormulaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes) 
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

with open(jsonpath, 'r') as f:# モデル読み込み
    class_names = json.load(f)
model = MathFormulaCNN(len(class_names))
model.load_state_dict(torch.load(modelpath, map_location=device))
model.to(device).eval()

def resize(img, size=(48, 48)):
    """認識モデルの入力用に文字を中央揃えで正方形にリサイズ"""
    # cv2.bitwise_not(img) 画素値を反転させ、白背景に黒文字の画像を「黒背景に白文字」へ変換。 
    # cv2.findContours() 文字の輪郭の集合を検出。  
    # cv2.boundingRect() 輪郭 c を囲む最小の矩形座標 (bx, by, bw, bh) を算出。  
    # np.full() 指定サイズ size で画素値255（白）の初期キャンバスを作成。  
    # cv2.resize() キャンバスをモデル指定のサイズに縮小。
    h, w = img.shape
    if h <= 0 or w <= 0: return np.full(size, 255, dtype=np.uint8)

    img_inv = cv2.bitwise_not(img)
    contours, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        xmin, ymin = w, h
        xmax, ymax = 0, 0
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            xmin, ymin = min(xmin, bx), min(ymin, by)
            xmax, ymax = max(xmax, bx + bw), max(ymax, by + bh)
        cropw, croph = xmax - xmin, ymax - ymin
        cropped = img[ymin:ymax, xmin:xmax]
    else:
        cropw, croph = w, h
        cropped = img.copy()

    maxside = max(cropw, croph)
    squareside = max(1, int(maxside * 2))

    canvas = np.full((squareside, squareside), 255, dtype=np.uint8)
    offsety = (squareside - croph) // 2
    offsetx = (squareside - cropw) // 2
    canvas[offsety:offsety + croph, offsetx:offsetx + cropw] = cropped

    return cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)

def roi_patch(patch):
    """切り出された文字画像をモデルに入力し認識結果を返す"""
    # transforms.Compose([]) 画像をテンソル化し、画素値を -1 〜 1 の範囲へ正規化する処理を順次実行。  
    # imgt = transform(roi_in).unsqueeze(0).to(device) 画像をモデル入力用に次元を拡張し、CPU/GPUデバイスへ転送。  
    # F.softmax(out, dim=1) モデルの出力を、合計1になる確率分布へ変換。  
    # torch.max(probs, 1) 最も高い確率を持つクラスIDと、その信頼度を取得。  
    roi_in = resize(patch, size=(48, 48))
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    imgt = transform(roi_in).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(imgt)
        probs = F.softmax(out, dim=1)
        confval, pred = torch.max(probs, 1)
        
    rawsym = class_names[pred.item()]
    char = symbolmap.get(rawsym, rawsym.replace('var_', ''))
    return char, rawsym, float(confval.item())

def merge_boxes(boxes, thres=15):
    """隣接するボックス同士を一つの大きなボックスに統合"""
    # curr[0], curr[1], curr[2], curr[3] それぞれ矩形の左上x座標、y座標、幅、高さ（x, y, w, h）を指す。  
    # overlap 2つの矩形が水平方向に重なっている画素数を計算。  
    # distx, disty 2つの矩形の間の水平・垂直方向の最短距離を計算。  
    # boxes.pop(0) / boxes.pop(i) 統合対象となる矩形リストから要素を取り出したり、削除したりして再帰的に処理。
    if not boxes: return []
    boxes = sorted(boxes, key=lambda x: x[0])
    merged = []
    
    while len(boxes) > 0:
        curr = list(boxes.pop(0))
        found = True
        while found:
            found = False
            for i in range(len(boxes)):
                next = boxes[i]
                overlap = max(0, min(curr[0]+curr[2], next[0]+next[2]) - max(curr[0], next[0]))
                distx = max(0, next[0] - (curr[0] + curr[2]), curr[0] - (next[0] + next[2]))
                disty = max(0, next[1] - (curr[1] + curr[3]), curr[1] - (next[1] + next[3]))

                isvertical = overlap > min(curr[2], next[2]) * 0.5
                vlimit = 25 if isvertical else thres
                
                if distx < 4 and disty < vlimit:
                    nx = min(curr[0], next[0])
                    ny = min(curr[1], next[1])
                    nw = max(curr[0]+curr[2], next[0]+next[2]) - nx
                    nh = max(curr[1]+curr[3], next[1]+next[3]) - ny
                    curr = [nx, ny, nw, nh]
                    boxes.pop(i)
                    found = True
                    break
        merged.append(tuple(curr))
    return merged

def detect(roi):
    """画像領域が分数であるかを判定し構成要素を分解"""
    # cv2.drawContours(mask, [c], -1, 255, -1) 輪郭 c の内部を255（白）で塗りつぶし、マスク画像を作成。  
    # np.full_like(roi, 255) 元の画像と同じサイズで、背景色が255の空画像を生成。  
    # charroi[mask == 255] = roi[mask == 255] マスク内の輪郭部分のみ、元の画素値をコピーして文字だけを切り出した画像を作成。
    h, w = roi.shape
    if h < 20 or w < 10: return None

    roi_inv = cv2.bitwise_not(roi)
    cnts, _ = cv2.findContours(roi_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) < 2: return None

    bestidx = -1
    maxlinew = 0
    linebox = None
    
    for i, c in enumerate(cnts):
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > w * 0.45 and bh < h * 0.5:
            if h * 0.15 < (by + bh/2) < h * 0.85:
                if bw > maxlinew:
                    maxlinew = bw
                    bestidx = i
                    linebox = (bx, by, bw, bh)
                    
    if bestidx == -1: return None 
        
    bx, by, bw, bh = linebox
    centery = by + bh // 2 
    
    numparts, denparts = [], []
    linepart = None
        
    for i, c in enumerate(cnts):
        cbx, cby, cbw, cbh = cv2.boundingRect(c)
        if cv2.contourArea(c) < 8: continue 
            
        mask = np.zeros_like(roi_inv)
        cv2.drawContours(mask, [c], -1, 255, -1)
        
        charroi = np.full_like(roi, 255) 
        charroi[mask == 255] = roi[mask == 255] 
        cropped = charroi[cby:cby+cbh, cbx:cbx+cbw]
        
        if i == bestidx:
            linepart = (cropped, (cbx, cby, cbw, cbh), 'line')
            continue

        char_centery = cby + cbh // 2
        if char_centery < centery:
            numparts.append((cropped, (cbx, cby, cbw, cbh), 'num'))
        else:
            denparts.append((cropped, (cbx, cby, cbw, cbh), 'den'))
            
    if not numparts or not denparts or linepart is None: return None
        
    numparts = sorted(numparts, key=lambda p: p[1][0])
    denparts = sorted(denparts, key=lambda p: p[1][0])
    return [linepart] + numparts + denparts

def preprocess_image(img):
    """前処理：傾き補正と適応的閾値による二値化"""
    # cvtColor	カラー画像を解析しやすいグレースケールに変換。
    # threshold	傾き検知用の「文字部分だけが白、背景が黒」のマスクを作成。
    # minAreaRect	文字の傾き角度を計算し、getRotationMatrix2D と warpAffine で水平に補正。
    # GaussianBlur	画像のノイズを低減。
    # adaptiveThreshold	影や照明ムラを無視し、文字部分を明確化する。

    if len(img.shape) == 3: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, temp = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(temp > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else (90 - angle if angle > 45 else -angle)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 45, 38)
    return img, thresh

def extract_boxes(thresh):
    """領域抽出：輪郭取得、マージ、分割、ボックス調整"""
    # cv2.bitwise_not() 入力画像の色を反転させる。  
    # cv2.findContours()　色を反転させた画像から輪郭情報を取得する。  
    # initboxes 輪郭の面積が15より大きいものだけを抽出し、その外接矩形を取得する。  
    # merge_boxes() 取得した矩形群統合。  
    # sorted(..., key=lambda x: x[0]) 統合後の矩形を、x座標が小さい順に並べ替る。  
    # if not tmpboxes: return []　矩形が一つも存在しない場合は空リストを返して終了する。  
    # avgw = np.mean()　全矩形の幅の平均値を算出します。  
    # if detect(roisplit)　矩形が分数と判定されず、かつ幅が平均幅の1.8倍以上、かつ高さの1.3倍以上ある場合に分割処理を行う。  
    # proj　分割候補矩形内のインク密度を列方向に合計する。  
    # splitrel 定範囲（幅の30%〜70%）内でインク密度が最も低い列を探し、分割位置を決定する。  
    # finalboxes.append()　分割位置で矩形を2つに分け、リストに追加。分割不要な場合はそのまま追加。
    # projx　各矩形内のインク密度を列方向に合計する。  
    # actualw　インクが存在する右端の列を特定し、矩形の幅をその位置まで切り詰める。  
    # limitl / limitr　隣接する矩形との境界を計算。  
    # x_adj, w_adj トリミングした幅を、隣接矩形と重ならない範囲内で適用する。  
    # if w_adj < h:　幅が高さより小さい場合、左右に余白を加えて、幅を高さと同じ値に引き伸す（アスペクト比補正）。  
    # adjboxes.append()　最終的な座標・サイズをリストに格納し、全ての矩形を返す。    
    cntimg = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(cntimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    initboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 15]
    tmpboxes = sorted(merge_boxes(initboxes, thres=6), key=lambda x: x[0])
    if not tmpboxes: return []

    avgw = np.mean([b[2] for b in tmpboxes])
    finalboxes = []
    for (x, y, w, h) in tmpboxes:#各矩形についての処理
        roisplit = thresh[y:y+h, x:x+w]
        if detect(roisplit) is None and w > avgw * 1.8 and w > h * 1.3:
            proj = np.sum(cv2.bitwise_not(roisplit), axis=0)
            s, e = int(w * 0.3), int(w * 0.7)
            splitrel = s + np.argmin(proj[s:e])
            finalboxes.append((x, y, splitrel, h))
            finalboxes.append((x + splitrel, y, w - splitrel, h))
        else: finalboxes.append((x, y, w, h))

    imgh, imgw = thresh.shape
    adjboxes = []
    for i, (x, y, w, h) in enumerate(finalboxes):#分割後の矩形群の処理。 
        roi_inv = cv2.bitwise_not(thresh[y:y+h, x:x+w]) 
        projx = np.sum(roi_inv, axis=0)         
        actualw = next((col + 1 for col in range(w - 1, -1, -1) if projx[col] > 0), w)
        
        limitl = 0 if i == 0 else adjboxes[i-1][0] + adjboxes[i-1][2]
        limitr = imgw if i == len(finalboxes) - 1 else finalboxes[i+1][0]
        x_adj, w_adj = max(x, limitl), max(1, min(actualw, limitr - max(x, limitl)))
        if w_adj < h: # アスペクト比の補正
            pad = (h - w_adj) // 2
            x_adj, w_adj = max(0, x_adj - pad), w_adj + (h - w_adj)
        adjboxes.append((x_adj, y, w_adj, h))
    return adjboxes

def recognize_characters(thresh, boxes):
    """認識：CNN推論と分数構造解析"""
    # rawresults 認識結果を格納するための空リストを作成。  
    # totalinv 画像全体を反転させ、輪郭検出用として保持。  
    # fracid = 0 分数ごとに付与する識別IDのカウンターを初期化。  
    # roi ボックス内の画像を切り出す。  
    # fracres 切り出した画像が分数構造（分子・分母・線）を含んでいるかを判定する。  
    # 　　　　 if ptype == 'line': パーツの種類が横線であれば \frac と判定し、そうでなければ roi_patch(croi) で文字を特定する。
    # rawresults.append() パーツの座標・認識文字・タイプ・所属分数ID等を辞書形式でリストに追加する。
    # subcnts, _ 　ボックス内の細かい輪郭情報を取得する。  
    # char, rawsym, conf　CNN推論により文字を特定する。
    # rawresults.append()　通常文字として、座標・文字・タイプ・ID等をリストに追加。  
    # return rawresults 全ボックス分の処理を終えた結果リストを返す。  
    rawresults, totalinv = [], cv2.bitwise_not(thresh)
    fracid = 0
    for x, y, w, h in boxes:
        roi = thresh[y:y+h, x:x+w]
        fracres = detect(roi)
        if fracres:#分数と判定された場合
            for croi, (sx, sy, sw, sh), ptype in fracres:
                absx, absy, absw, absh = x + sx, y + sy, sw, sh
                char, rawsym, conf = ('\\frac', 'frac_line', 1.0) if ptype == 'line' else roi_patch(croi)
                if char == '1' and ptype != 'line' and len(cv2.findContours(cv2.bitwise_not(croi), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) >= 2: char = 'i'
                rawresults.append({'char': char, 'box': [absx, absy, absw, absh], 'conf': conf, 'ptype': ptype, 'linebox': [absx, absy, absw, absh] if ptype == 'line' else None, 'rawsym': rawsym, 'fracid': fracid})
            fracid += 1
        else:#分数ではない場合
            # roiの詳細解析 
            roi_inv = totalinv[y:y+h, x:x+w]
            subcnts, _ = cv2.findContours(roi_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            char, rawsym, conf = roi_patch(roi)
            if char == '1' and len([c for c in subcnts if cv2.contourArea(c) >= 2]) >= 2: char = 'i'#文字が「1」かつ輪郭数が2つ以上ある場合、英字の「i」に置き換えます。
            rawresults.append({'char': char, 'box': [x, y, w, h], 'conf': conf, 'ptype': None, 'linebox': None, 'rawsym': rawsym, 'fracid': None})
    return rawresults

def postprocess_results(rawresults, thresh):
    """【工程4】後処理：記号結合、整形、Latex変換"""
    # if cur['rawsym'] == 'r_sp' 現在のパーツが空白や連結対象の記号で、かつ次のパーツとの距離が15px未満の場合、2つのパーツを1つの矩形にまとめ直す。  
    # newchar, ... = roi_patch(...) 結合した矩形画像を再度CNNに渡し、小文字のxかどうかチェック。
    # if res['rawsym'] == 'var_c' ... 変数やカッコ候補を判定し、後方に ) がある場合など、必要に応じてcを ( に書き換る。  
    # if res['char'] == '1' ...：数字の「1」が後続のパーツより高さで1.3倍以上ある場合、積分記号 \int  に置き換る。
    # numstr / denstr　同じ分数グループ内の num（分子）と den（分母）を、x座標順に結合して文字列化する。  
    # reconresults.append()　分子・分母・横線をまとめて1つの \frac{...}{...} 文字列として reconresults に格納。  
    # is_super 判定：パーツの垂直位置（y座標）が前のパーツより一定以上高い場合、上付き文字（べき乗など）と判断。  
    # char = "^" + char　上付き文字であれば、文字の先頭に ^ を付与。  
    # fulltext = "".join(reslist)　すべてのパーツを1つの文字列に繋げる。  
    # fulltext.replace()　辞書で定義された誤認識の組み合わせ（例：「s1n」→「\sin 」）を正しいLaTeX関数に置換。  
    # return fulltext, details　最終的なLaTeX文字列と、各文字の座標情報を含む詳細リストを返す。  
    idx = 0
    while idx < len(rawresults) - 1:#隣接する2つのパーツを比較。(小文字のxが手書きでは隙間ができる可能性がある)
        cur, nxt = rawresults[idx], rawresults[idx + 1]
        if cur['rawsym'] == 'r_sp' and (idx == 0 or rawresults[idx - 1]['rawsym'] != 'var_c') and cur['fracid'] == nxt['fracid']:
            box1, box2 = cur['box'], nxt['box']
            if box2[0] - (box1[0] + box1[2]) < 15:
                mx, my = min(box1[0], box2[0]), min(box1[1], box2[1])
                mw, mh = max(box1[0] + box1[2], box2[0] + box2[2]) - mx, max(box1[1] + box1[3], box2[1] + box2[3]) - my
                newchar, newraw, newconf = roi_patch(thresh[my:my+mh, mx:mx+mw])
                rawresults[idx] = {**cur, 'char': newchar, 'box': [mx, my, mw, mh], 'conf': newconf, 'rawsym': newraw}
                rawresults.pop(idx + 1); continue
        idx += 1
    # カッコと積分修正
    for i, res in enumerate(rawresults):
        if res['rawsym'] == 'var_c' and not (i + 2 < len(rawresults) and res['char'] in ['o', '0'] and rawresults[i+2]['char'] in ['s', '1', '5']):
            if any(r['rawsym'] == 'r_sp' or r['char'] == ')' for r in rawresults[i+1:]): res['char'] = '('
        if res['char'] == '1' and i + 1 < len(rawresults) and res['box'][3] >= rawresults[i+1]['box'][3] * 1.3: res['char'] = '\\int '
    
    # LaTeX構築
    reconresults, processedfracs = [], set()
    for res in rawresults:#分数構造の再構築し全パーツをグループ化。
        fid = res['fracid']
        if fid is None: reconresults.append(res); continue
        if fid in processedfracs: continue
        group = [r for r in rawresults if r['fracid'] == fid]
        numstr = "".join([r['char'] for r in sorted([r for r in group if r['ptype'] == 'num'], key=lambda r: r['box'][0])])
        denstr = "".join([r['char'] for r in sorted([r for r in group if r['ptype'] == 'den'], key=lambda r: r['box'][0])])
        reconresults.append({'char': f"\\frac{{{numstr}}}{{{denstr}}}", 'box': next((r['box'] for r in group if r['ptype'] == 'line'), group[0]['box']), 'conf': np.mean([r['conf'] for r in group]), 'ptype': 'reconstructed_frac'})
        processedfracs.add(fid)
    
    reslist, details, prev = [], [], None
    for res in reconresults:#べき乗の判定
        char, (x, y, w, h) = res['char'], res['box']
        is_super = False
        if prev:
            px, py, pw, ph, pt, pl = prev
            is_super = (y + h < pl[1] - (pl[3] * 1.5)) if (pt == 'den' and pl) else (y + h < py + ph * 0.6)
        if is_super and reslist and reslist[-1] not in ['(', '\\frac', '=', '+', '-']: char = "^" + char
        reslist.append(char); details.append({'char': char, 'box': [x, y, w, h], 'conf': res['conf']})
        prev = (x, y, w, h, res['ptype'], res.get('linebox'))
        
    fulltext = "".join(reslist)#置換マップによる修正と出力
    for orig, rep in {"s1n": "\\sin ", "5in": "\\sin ", "c0s": "\\cos ", "+an": "\\tan "}.items(): fulltext = fulltext.replace(orig, rep)
    return fulltext, details

def predict_formula_main(img):
    """メイン関数"""
    # predict_formula_main 関数の処理ステップ画像の準備
    # preprocess_image() 入力画像に対し、傾き補正や二値化を行い、解析可能な状態に整える。（img と thresh）  
    # extract_boxes()　前処理済みの画像から、文字や数式パーツが含まれる矩形領域を抽出・補正。  
    # if not boxes: return "", [] 抽出された領域がない場合は、空の結果を返して処理を終了。  
    # recognize_characters() 抽出された各矩形領域に対してCNNによる推論や分数構造の判定を行い、文字パーツのリストを作成する。（rawresults）  
    # postprocess_results() 認識結果を統合・整形し、LaTeX形式の文字列と詳細な座標情報を生成して返す。  
    img, thresh = preprocess_image(img)#画像の準備
    boxes = extract_boxes(thresh)#領域の抽出
    if not boxes: return "", []
    rawresults = recognize_characters(thresh, boxes)#文字の認識
    return postprocess_results(rawresults, thresh)#LaTeX形式への変換