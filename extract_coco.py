#!/usr/bin/env python3
"""
Extract COCO 2017 dataset ZIP files to proper directory structure
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path

def extract_zip(zip_path, extract_to):
    """Extract ZIP file with progress"""
    print(f"\n⬜ Extracting {Path(zip_path).name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract COCO 2017 dataset')
    parser.add_argument(
        '--data-dir',
        default='../coco',
        help='COCO dataset directory (default: ../coco)'
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        sys.exit(1)
    
    print("="*70)
    print("📦 COCO 2017 Dataset Extraction")
    print("="*70)
    
    # Define extractions
    extractions = [
        (data_dir / 'train2017.zip', data_dir / 'images'),
        (data_dir / 'val2017.zip', data_dir / 'images'),
        (data_dir / 'test2017.zip', data_dir / 'images'),
        (data_dir / 'annotations_trainval2017.zip', data_dir / 'annotations'),
        (data_dir / 'image_info_test2017.zip', data_dir / 'annotations'),
    ]
    
    success_count = 0
    for zip_path, extract_to in extractions:
        if zip_path.exists():
            extract_to.mkdir(parents=True, exist_ok=True)
            if extract_zip(str(zip_path), str(extract_to)):
                success_count += 1
        else:
            print(f"⏭️  Skipping (not found): {zip_path.name}")
    
    print("\n" + "="*70)
    print(f"✅ Extraction complete! ({success_count} file(s) extracted)")
    print("\nFinal structure:")
    print("  ../coco/")
    print("  ├── images/")
    print("  │   ├── train2017/  (118,287 images)")
    print("  │   ├── val2017/    (5,000 images)")
    print("  │   └── test2017/   (40,670 images)")
    print("  └── annotations/")
    print("      ├── instances_train2017.json")
    print("      ├── instances_val2017.json")
    print("      └── image_info_test2017.json")
    print("\nReady to run evaluation:")
    print("  python tools/eval.py --data data/coco.yaml --weights weights/yolov6s.pt --task val")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
