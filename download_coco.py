#!/usr/bin/env python3
"""
COCO 2017 Dataset Downloader for YOLOv6

Downloads COCO 2017 train/val/test splits (~19GB total).
Supports resuming interrupted downloads.

Usage:
    python download_coco.py              # Downloads all splits
    python download_coco.py --split train  # Download only train
    python download_coco.py --split val    # Download only val
"""

import os
import sys
import urllib.request
import argparse
from pathlib import Path
import hashlib

# COCO 2017 dataset URLs
COCO_URLS = {
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'test_images': 'http://images.cocodataset.org/zips/test2017.zip',
    'train_anno': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    'test_anno': 'http://images.cocodataset.org/annotations/image_info_test2017.zip',
}

# Sizes (for reference)
SIZES = {
    'train_images': '13.5 GB',
    'val_images': '0.8 GB',
    'test_images': '0.9 GB',
    'train_anno': '252 MB',
    'test_anno': '20 MB',
}

class ProgressBar:
    """Simple progress bar for downloads"""
    def __init__(self, total):
        self.total = total
        self.current = 0
        
    def update(self, chunk_size):
        self.current += chunk_size
        percent = (self.current / self.total) * 100
        bar_length = 40
        filled = int(bar_length * self.current / self.total)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\rDownloading: |{bar}| {percent:.1f}% ({self.current/1e9:.1f}GB/{self.total/1e9:.1f}GB)', end='')

def download_file(url, output_path, resume=True):
    """Download a file with resume support and progress bar"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if output_path.exists():
        print(f"✓ {output_path.name} already exists, skipping download")
        return True
    
    print(f"\n⬇️  Downloading {output_path.name}...")
    print(f"   URL: {url}")
    
    try:
        req = urllib.request.Request(url)
        
        # Resume if file partially downloaded
        if output_path.exists():
            file_size = output_path.stat().st_size
            req.add_header('Range', f'bytes={file_size}-')
            mode = 'ab'
        else:
            mode = 'wb'
        
        # Get total size
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            if total_size == 0:
                # If server doesn't support Content-Length, just download
                with output_path.open(mode) as f:
                    f.write(response.read())
                print(f"✓ Downloaded to {output_path}")
                return True
            
            # Download with progress bar
            progress = ProgressBar(total_size)
            with output_path.open(mode) as f:
                while True:
                    chunk = response.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(len(chunk))
            
            print(f"\n✓ Downloaded to {output_path}")
            return True
            
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Download COCO 2017 dataset for YOLOv6',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_coco.py                    # Download all splits
  python download_coco.py --split train      # Train images + annotations
  python download_coco.py --split val        # Val images + annotations
  python download_coco.py --split test       # Test images + annotations
        """
    )
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test', 'all'],
        default='all',
        help='Dataset split to download (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        default='../coco',
        help='Path to save COCO dataset (default: ../coco)'
    )
    parser.add_argument(
        '--no-annotations',
        action='store_true',
        help='Skip downloading annotations'
    )
    args = parser.parse_args()
    
    # Define what to download based on split
    downloads = {}
    
    if args.split in ['train', 'all']:
        downloads['train_images'] = os.path.join(args.data_dir, 'train2017.zip')
        if not args.no_annotations:
            downloads['train_anno'] = os.path.join(args.data_dir, 'annotations_trainval2017.zip')
    
    if args.split in ['val', 'all']:
        downloads['val_images'] = os.path.join(args.data_dir, 'val2017.zip')
        if not args.no_annotations and 'train_anno' not in downloads:
            downloads['train_anno'] = os.path.join(args.data_dir, 'annotations_trainval2017.zip')
    
    if args.split in ['test', 'all']:
        downloads['test_images'] = os.path.join(args.data_dir, 'test2017.zip')
        if not args.no_annotations:
            downloads['test_anno'] = os.path.join(args.data_dir, 'image_info_test2017.zip')
    
    print("\n" + "="*70)
    print("🚀 COCO 2017 Dataset Downloader")
    print("="*70)
    print(f"\nDataset split: {args.split}")
    print(f"Save location: {args.data_dir}")
    print(f"\nTotal files to download: {len(downloads)}")
    total_size_str = " + ".join([SIZES.get(k, "unknown") for k in downloads.keys()])
    print(f"Estimated total size: {total_size_str}")
    print(f"Note: This will take 30min-2hrs depending on your connection.")
    print("\n" + "="*70)
    
    # Download files
    failed = []
    for key, path in downloads.items():
        url = COCO_URLS.get(key)
        if not url:
            print(f"⚠️  Unknown key: {key}")
            continue
        
        success = download_file(url, path)
        if not success:
            failed.append(key)
    
    # Summary
    print("\n" + "="*70)
    if failed:
        print(f"❌ {len(failed)} download(s) failed: {', '.join(failed)}")
        print("   Retry by running the script again (supports resume)")
        sys.exit(1)
    else:
        print("✅ All downloads completed!")
        print("\nNext steps:")
        print("  1. Extract ZIP files manually (or use extract_coco.py)")
        print("  2. Run evaluation:")
        print("     python tools/eval.py --data data/coco.yaml --weights weights/yolov6s.pt --task val")
        print("\nDataset structure:")
        print("  ../coco/")
        print("  ├── images/")
        print("  │   ├── train2017/")
        print("  │   ├── val2017/")
        print("  │   └── test2017/")
        print("  └── annotations/")
        print("      ├── instances_train2017.json")
        print("      ├── instances_val2017.json")
        print("      └── image_info_test2017.json")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
