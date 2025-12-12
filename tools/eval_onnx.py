import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
 
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while maintaining aspect ratio"""
    shape = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)
 
class ONNXEvaluator:
    def __init__(self, onnx_path, data_dir, json_path):
        # Use Intel GPU (Dml) if available, else CPU
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.data_dir = Path(data_dir)
        self.coco = COCO(json_path)
        self.img_ids = self.coco.getImgIds()
        
    def detect(self, img_path):
        img_raw = cv2.imread(str(img_path))
        if img_raw is None: return []
        
        # Preprocess
        img_in, ratio, (dw, dh) = letterbox(img_raw)
        img_in = img_in.transpose((2, 0, 1))[::-1] # HWC -> CHW, BGR -> RGB
        img_in = np.ascontiguousarray(img_in) / 255.0
        outputs = self.session.run(None, {self.input_name: img_in[None].astype(np.float32)})
        
        # Filter Raw Boxes (NMS)
        rows = outputs[0][0] # Shape: [8400, 85]
        boxes, scores, class_ids = [], [], []
        
        # Optimization: Only process rows with object confidence > 0.001
        valid_rows = rows[rows[:, 4] > 0.05]
        
        for row in valid_rows:
            obj_conf = row[4]
            class_scores = row[5:]
            class_id = np.argmax(class_scores)
            score = obj_conf * class_scores[class_id]
            
            if score > 0.05:
                cx, cy, w, h = row[0:4]
                # Scale back to original image
                boxes.append([(cx - w/2 - dw)/ratio, (cy - h/2 - dh)/ratio, w/ratio, h/ratio])
                scores.append(float(score))
                class_ids.append(int(class_id))
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.05, 0.65)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                # Map standard YOLO class ID (0-79) to COCO ID (1-90)
                coco_cats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
                cat_id = coco_cats[class_ids[i]] if class_ids[i] < len(coco_cats) else class_ids[i]
                
                results.append({
                    "image_id": 0, # Will be filled by run()
                    "category_id": cat_id,
                    "bbox": [float(x) for x in boxes[i]],
                    "score": scores[i]
                })
        return results
 
    def run(self):
        results = []
        print(f"🚀 Evaluating ONNX model on {len(self.img_ids)} images...")
        for img_id in tqdm(self.img_ids):
            info = self.coco.loadImgs(img_id)[0]
            file_name = info['file_name']
            dets = self.detect(self.data_dir / file_name)
            for d in dets:
                d['image_id'] = img_id
                results.append(d)
                
        if results:
            with open("results.json", "w") as f: json.dump(results, f)
            print("\n📊 Calculating mAP...")
            cocoDt = self.coco.loadRes("results.json")
            cocoEval = COCOeval(self.coco, cocoDt, "bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
        else:
            print("❌ No detections found.")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--json', required=True)
    args = parser.parse_args()
    ONNXEvaluator(args.onnx, args.data, args.json).run()
 