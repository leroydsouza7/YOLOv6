# YOLOv6 Inference & Evaluation Guide

## ✅ Environment Setup Complete

Your YOLOv6 Python 3.8 environment is fully configured with all required dependencies:
- Python 3.8.20
- PyTorch 2.4.1
- TorchVision 0.20.0
- OpenCV 4.12.0
- ONNX 1.17.0
- All other requirements from `requirements.txt`

### How to Activate the Environment

```powershell
# Load conda hook and activate yolov6 env
& 'C:\Users\leroy.dsouza\AppData\Local\anaconda3\shell\condabin\conda-hook.ps1' ; conda activate yolov6

# Or in cmd.exe
C:\Users\leroy.dsouza\AppData\Local\anaconda3\Scripts\activate.bat yolov6
```

---

## 🎯 Inference Examples

### 1. Basic Inference on Images

```powershell
python tools/infer.py `
  --weights weights/yolov6s.pt `
  --source data/images `
  --device cpu `
  --conf-thres 0.4 `
  --save-dir runs/inference `
  --name exp_basic
```

**Output:** Detection results saved to `runs/inference/images/`

### 2. Inference with Custom Confidence Threshold

```powershell
python tools/infer.py `
  --weights weights/yolov6s.pt `
  --source data/images `
  --device cpu `
  --conf-thres 0.5 `
  --iou-thres 0.5 `
  --save-dir runs/inference `
  --name exp_high_conf
```

### 3. Save Detection Results as Text

```powershell
python tools/infer.py `
  --weights weights/yolov6s.pt `
  --source data/images `
  --device cpu `
  --save-txt `
  --save-dir runs/inference `
  --name exp_with_txt
```

**Output:** 
- Images with detections: `runs/inference/images/`
- Text files with detections: `runs/inference/labels/`

### 4. Single Image Inference

```powershell
python tools/infer.py `
  --weights weights/yolov6s.pt `
  --source data/images/image1.jpg `
  --device cpu `
  --save-dir runs/inference `
  --name single_image
```

### 5. Inference on Different Model Sizes

```powershell
# Nano (fastest, ~4.7M params)
python tools/infer.py --weights weights/yolov6n.pt --source data/images --device cpu

# Small (balanced, ~18.5M params) - RECOMMENDED for most cases
python tools/infer.py --weights weights/yolov6s.pt --source data/images --device cpu

# Medium (higher accuracy, ~34.9M params)
python tools/infer.py --weights weights/yolov6m.pt --source data/images --device cpu

# Large (best accuracy, ~59.6M params)
python tools/infer.py --weights weights/yolov6l.pt --source data/images --device cpu
```

---

## 📊 Evaluation & Benchmarking

### 1. Speed Benchmark (FLOPs and Inference Speed)

```powershell
python tools/eval.py `
  --weights weights/yolov6s.pt `
  --device cpu `
  --task speed `
  --img-size 640
```

### 2. Evaluation on Your Own Dataset

First, prepare your dataset in YOLO format and create a YAML file (`data/custom.yaml`):

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 80  # number of classes
names: ['person', 'bicycle', 'car', ...]  # class names
```

Then run:

```powershell
python tools/eval.py `
  --data data/custom.yaml `
  --weights weights/yolov6s.pt `
  --batch-size 32 `
  --img-size 640 `
  --device cpu `
  --task val `
  --save_dir runs/val `
  --name my_eval
```

### 3. Evaluation with Metrics

```powershell
python tools/eval.py `
  --data data/custom.yaml `
  --weights weights/yolov6s.pt `
  --batch-size 16 `
  --img-size 640 `
  --device cpu `
  --task val `
  --do_coco_metric True `
  --do_pr_metric True `
  --plot_curve True `
  --save_dir runs/val `
  --name detailed_eval
```

---

## 📥 Downloading Pre-trained Models

Use the provided `download_model.py` script:

```powershell
# Download YOLOv6 Nano
python download_model.py n

# Download YOLOv6 Small
python download_model.py s

# Download YOLOv6 Medium
python download_model.py m

# Download YOLOv6 Large
python download_model.py l
```

Models will be saved to `weights/yolov6{size}.pt`

---

## 🎛️ Key Parameters Reference

### Inference Parameters (`tools/infer.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--weights` | `weights/yolov6s.pt` | Path to model weights |
| `--source` | `data/images` | Input image/directory path |
| `--device` | `0` | Device to use: 'cpu' or GPU ID (0,1,2,3) |
| `--conf-thres` | `0.4` | Confidence threshold (0-1) |
| `--iou-thres` | `0.45` | NMS IoU threshold (0-1) |
| `--img-size` | `640 640` | Input image size (height width) |
| `--max-det` | `1000` | Maximum detections per image |
| `--save-txt` | False | Save detection results as .txt files |
| `--save-dir` | `runs/inference` | Output directory |
| `--name` | `exp` | Experiment name (creates subdirectory) |

### Evaluation Parameters (`tools/eval.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | `data/coco.yaml` | Dataset YAML file path |
| `--weights` | `weights/yolov6s.pt` | Model weights path |
| `--batch-size` | `32` | Batch size for evaluation |
| `--img-size` | `640` | Evaluation image size |
| `--conf-thres` | `0.03` | Confidence threshold |
| `--iou-thres` | `0.65` | NMS IoU threshold |
| `--device` | `0` | Device: 'cpu' or GPU ID |
| `--task` | `val` | Task: 'val', 'test', or 'speed' |
| `--half` | False | Use FP16 half-precision |
| `--save_dir` | `runs/val/` | Results save directory |
| `--name` | `exp` | Experiment name |

---

## 🚀 Quick Tips

### For GPU Acceleration (Recommended)
```powershell
# If you have CUDA 11.8 compatible GPU, use GPU device 0
python tools/infer.py --weights weights/yolov6s.pt --source data/images --device 0
```

### For Batch Processing Multiple Images
```powershell
# Use a directory as source (automatically processes all images)
python tools/infer.py --weights weights/yolov6s.pt --source /path/to/images/folder --device cpu
```

### For High-Speed Inference with Lower Accuracy
```powershell
python tools/infer.py `
  --weights weights/yolov6n.pt `
  --source data/images `
  --device cpu `
  --conf-thres 0.5 `
  --img-size 416
```

### For Best Accuracy (Slower Inference)
```powershell
python tools/infer.py `
  --weights weights/yolov6l.pt `
  --source data/images `
  --device 0 `
  --conf-thres 0.1 `
  --img-size 1280
```

---

## 📁 Output Directories

- **Inference results:** `runs/inference/exp_name/images/`
- **Text detections:** `runs/inference/exp_name/labels/`
- **Evaluation results:** `runs/val/exp_name/`
- **Metrics plots:** `runs/val/exp_name/` (PR curve, confusion matrix, etc.)

---

## 🔗 Additional Resources

- **Official YOLOv6 GitHub:** https://github.com/meituan/YOLOv6
- **Model Architecture Details:** Check `yolov6/models/`
- **Training Guide:** Check `docs/Train_custom_data.md`
- **Quantization Guide:** Check `docs/Tutorial of Quantization.md`

---

## ✨ What We've Done

1. ✅ Set up Python 3.8 conda environment with all dependencies
2. ✅ Downloaded YOLOv6s pre-trained model (weights/yolov6s.pt)
3. ✅ Ran successful inference on sample images (results in `runs/inference/`)
4. ✅ Created helper scripts (`download_model.py`, `run_examples.py`)
5. ✅ Provided comprehensive command reference above

**You're ready to use YOLOv6 for inference and evaluation!** 🎉

---

## Next Steps

1. **Try Different Models:** Download and test different model sizes (n, m, l)
2. **Prepare Your Own Data:** Follow the dataset preparation guide in `docs/Train_custom_data.md`
3. **Fine-tune on Custom Data:** Use `tools/train.py` to train on your dataset
4. **Deploy Models:** Check `deploy/` folder for ONNX, TensorRT, and other deployment options

