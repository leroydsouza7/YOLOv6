#!/usr/bin/env python3
"""
Quick Start Guide for YOLOv6 Inference and Evaluation

This script demonstrates how to run inference and evaluation with YOLOv6.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the description"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode

def main():
    print("\n" + "="*70)
    print("YOLOv6 Inference & Evaluation Quick Start")
    print("="*70)
    
    # Example 1: Basic Inference
    print("\n📋 EXAMPLE 1: Basic Inference on Images")
    print("-" * 70)
    cmd1 = (
        "python tools/infer.py "
        "--weights weights/yolov6s.pt "
        "--source data/images "
        "--device cpu "
        "--conf-thres 0.4 "
        "--save-dir runs/inference "
        "--name exp_basic"
    )
    print(f"Command:\n{cmd1}\n")
    
    # Example 2: Inference with Confidence and NMS thresholds
    print("\n📋 EXAMPLE 2: Inference with Custom Thresholds")
    print("-" * 70)
    cmd2 = (
        "python tools/infer.py "
        "--weights weights/yolov6s.pt "
        "--source data/images "
        "--device cpu "
        "--conf-thres 0.5 "
        "--iou-thres 0.5 "
        "--save-dir runs/inference "
        "--name exp_custom"
    )
    print(f"Command:\n{cmd2}\n")
    
    # Example 3: Save detection results as text
    print("\n📋 EXAMPLE 3: Save Detection Results as Text")
    print("-" * 70)
    cmd3 = (
        "python tools/infer.py "
        "--weights weights/yolov6s.pt "
        "--source data/images "
        "--device cpu "
        "--save-txt "
        "--save-dir runs/inference "
        "--name exp_with_txt"
    )
    print(f"Command:\n{cmd3}\n")
    
    # Example 4: Evaluation (if you have a dataset with annotations)
    print("\n📋 EXAMPLE 4: Evaluation on a Dataset")
    print("-" * 70)
    cmd4 = (
        "python tools/eval.py "
        "--data data/coco.yaml "
        "--weights weights/yolov6s.pt "
        "--batch-size 16 "
        "--img-size 640 "
        "--device cpu "
        "--task val "
        "--save_dir runs/val "
        "--name exp_eval"
    )
    print(f"Command:\n{cmd4}\n")
    
    # Example 5: Speed benchmark
    print("\n📋 EXAMPLE 5: Speed/Performance Benchmark")
    print("-" * 70)
    cmd5 = (
        "python tools/eval.py "
        "--data data/coco.yaml "
        "--weights weights/yolov6s.pt "
        "--device cpu "
        "--task speed "
        "--img-size 640"
    )
    print(f"Command:\n{cmd5}\n")
    
    print("\n" + "="*70)
    print("📖 QUICK REFERENCE - Main Parameters")
    print("="*70)
    print("""
    Inference Parameters:
    ├── --weights: Path to model weights (e.g., weights/yolov6s.pt)
    ├── --source: Input source (image/dir/video) (e.g., data/images)
    ├── --device: 'cpu' or GPU device ID (e.g., 0, or 0,1,2,3)
    ├── --conf-thres: Confidence threshold [0-1] (default: 0.4)
    ├── --iou-thres: NMS IoU threshold [0-1] (default: 0.45)
    ├── --img-size: Input image size (default: 640 640)
    ├── --save-txt: Save detection results as text files
    ├── --save-dir: Directory to save results
    └── --name: Experiment name (creates subdirectory)

    Evaluation Parameters:
    ├── --data: Path to dataset YAML (e.g., data/coco.yaml)
    ├── --weights: Path to model weights
    ├── --batch-size: Batch size for evaluation (default: 32)
    ├── --img-size: Evaluation image size (default: 640)
    ├── --task: 'val', 'test', or 'speed'
    ├── --device: 'cpu' or GPU device ID
    └── --half: Use FP16 half-precision for speed

    Model Sizes Available:
    ├── yolov6n.pt (nano, ~4.7M parameters)
    ├── yolov6s.pt (small, ~18.5M parameters)
    ├── yolov6m.pt (medium, ~34.9M parameters)
    └── yolov6l.pt (large, ~59.6M parameters)
    """)
    
    print("\n" + "="*70)
    print("✨ Running Examples (uncomment to execute)")
    print("="*70)
    
    # Uncomment the example you want to run:
    # run_command(cmd1, "Running Basic Inference")
    # run_command(cmd2, "Running Inference with Custom Thresholds")
    # run_command(cmd3, "Running Inference and Saving Text Results")
    # run_command(cmd4, "Running Evaluation")
    # run_command(cmd5, "Running Speed Benchmark")
    
    print("\n✅ To run an example, uncomment it in the examples above and execute:")
    print("   python run_examples.py\n")

if __name__ == '__main__':
    main()
