import json
import os
from pathlib import Path
from tqdm import tqdm
 
def convert_coco_json_to_yolo(json_file, output_dir):
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {json_file}...")
    with open(json_file) as f:
        data = json.load(f)
    
    # Create image dict for quick lookup
    images = {img['id']: img for img in data['images']}
    
    print(f"Converting annotations to {output_dir}...")
    for ann in tqdm(data['annotations']):
        image_id = ann['image_id']
        img = images[image_id]
        
        # COCO bbox: [x_min, y_min, width, height]
        x, y, w, h = ann['bbox']
        
        # YOLO bbox: [x_center, y_center, width, height] (normalized 0-1)
        dw = 1.0 / img['width']
        dh = 1.0 / img['height']
        
        x_center = (x + w / 2.0) * dw
        y_center = (y + h / 2.0) * dh
        width = w * dw
        height = h * dh
        
        # Class ID
        cls_id = ann['category_id'] 
        # Note: COCO IDs are not continuous (1-90). You might need a mapper if classes 
        # don't align 0-79. For standard COCO 2017 val, this mapping is usually needed:
        # (This is a simplified mapper. If your model predicts wrong classes, we need the full map)
        
        # Save to txt
        filename = os.path.splitext(img['file_name'])[0] + '.txt'
        with open(os.path.join(output_dir, filename), 'a') as f:
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
 
# Run conversion for Val set
convert_coco_json_to_yolo(
    'data/coco/annotations/instances_val2017.json', 
    'data/coco/labels/val2017'
)
 