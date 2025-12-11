#!/usr/bin/env python3
"""
Simple script to download YOLOv6 pre-trained models from GitHub releases.
Run: python download_model.py [model_size]
Example: python download_model.py s    # downloads yolov6s.pt
"""

import os
import urllib.request
import sys

# YOLOv6 model download URLs (from official YOLOv6 repository)
MODELS = {
    'n': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt',
    's': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt',
    'm': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m.pt',
    'l': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l.pt',
}

def download_model(model_size='s'):
    """Download YOLOv6 pre-trained model"""
    if model_size not in MODELS:
        print(f"Error: Model size must be one of {list(MODELS.keys())}")
        sys.exit(1)
    
    url = MODELS[model_size]
    model_name = f'yolov6{model_size}.pt'
    weights_dir = 'weights'
    
    # Create weights directory if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)
    
    model_path = os.path.join(weights_dir, model_name)
    
    if os.path.exists(model_path):
        print(f"Model {model_name} already exists at {model_path}")
        return model_path
    
    print(f"Downloading {model_name} from {url}...")
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"Successfully downloaded {model_name} to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == '__main__':
    model_size = sys.argv[1] if len(sys.argv) > 1 else 's'
    download_model(model_size)
