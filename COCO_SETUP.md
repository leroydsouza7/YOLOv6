# COCO 2017 Dataset Setup for YOLOv6

## Quick Start

Download and set up COCO 2017 for YOLOv6 evaluation in 3 steps:

### Step 1: Download COCO Dataset (Recommended: Start with val split)

```powershell
# Download validation set only (~1.8 GB) - fastest to test
python download_coco.py --split val

# Or download all splits (train + val + test, ~19 GB)
python download_coco.py --split all
```

**What gets downloaded:**
- `val2017.zip` (5,000 images, 0.8 GB)
- `annotations_trainval2017.zip` (includes val annotations, 252 MB)
- Optionally: train, test images and annotations

Download time estimates:
- Val only: 10-30 min (1.8 GB)
- All splits: 1-3 hours (19 GB)

### Step 2: Extract Downloaded ZIPs

```powershell
python extract_coco.py
```

This creates the directory structure:
```
../coco/
├── images/
│   ├── train2017/     (118,287 images)
│   ├── val2017/       (5,000 images)
│   └── test2017/      (40,670 images)
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    └── image_info_test2017.json
```

### Step 3: Run Evaluation

```powershell
# Validate YOLOv6s on COCO val2017
python tools/eval.py `
  --data data/coco.yaml `
  --weights weights/yolov6s.pt `
  --batch-size 32 `
  --img-size 640 `
  --device cpu `
  --task val

# With GPU (if available)
python tools/eval.py `
  --data data/coco.yaml `
  --weights weights/yolov6s.pt `
  --batch-size 32 `
  --device 0 `
  --task val
```

---

## Evaluation Examples

### Basic Validation (Val Set)

```powershell
python tools/eval.py `
  --data data/coco.yaml `
  --weights weights/yolov6s.pt `
  --task val `
  --save_dir runs/val `
  --name my_eval
```

**Output:**
- mAP@0.5, mAP@0.75, mAP@0.5:0.95 metrics
- Per-class results
- Results saved to `runs/val/my_eval/`

### Validation with Detailed Metrics

```powershell
python tools/eval.py `
  --data data/coco.yaml `
  --weights weights/yolov6s.pt `
  --task val `
  --do_coco_metric True `
  --do_pr_metric True `
  --plot_curve True `
  --verbose
```

### Speed Benchmark

```powershell
python tools/eval.py `
  --data data/coco.yaml `
  --weights weights/yolov6s.pt `
  --task speed `
  --img-size 640
```

### Test Set Inference

```powershell
python tools/eval.py `
  --data data/coco.yaml `
  --weights weights/yolov6s.pt `
  --task test
```

---

## COCO Dataset Info

**COCO 2017 Splits:**
- **Train2017**: 118,287 images, 80 object classes
- **Val2017**: 5,000 images (used for validation/benchmarking)
- **Test2017**: 40,670 images (used for challenge submissions)

**Size Reference:**
- Train images: 13.5 GB
- Val images: 0.8 GB
- Test images: 0.9 GB
- Annotations: 272 MB total
- **Total: ~19 GB** (for all splits)

**80 Classes:**
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

---

## Alternative: Use Official Script

COCO also provides download scripts:

```powershell
# Clone COCO tools repo
git clone https://github.com/cocodataset/cocoapi.git

# Navigate to tools
cd cocoapi/PythonAPI

# Run their downloader (requires wget or curl)
python ../downloader.py --download_dir ../coco
```

---

## Troubleshooting

### Download Interrupted?
Resume automatically — just run `download_coco.py` again. It skips files that already exist.

### Not Enough Disk Space?
Download only validation split:
```powershell
python download_coco.py --split val  # ~1.8 GB
```

### Path Issues?
Update `data/coco.yaml` to point to your COCO location:
```yaml
train: d:\my_data\coco\images\train2017  # Windows
val: d:\my_data\coco\images\val2017
```

### Slow Downloads?
- Use a faster connection or mirror
- Download during off-peak hours
- Consider downloading only val split first

### Evaluation Errors?
- Ensure paths in `coco.yaml` are correct
- Verify ZIP extraction was complete
- Check that annotations JSON files exist

---

## Performance Benchmarks (YOLOv6 on COCO val2017)

| Model | mAP@0.5:0.95 | Speed (T4 GPU) | Params |
|-------|--------------|----------------|--------|
| YOLOv6-N | 37.5% | 779 fps | 4.7M |
| YOLOv6-S | 45.0% | 339 fps | 18.5M |
| YOLOv6-M | 50.0% | 175 fps | 34.9M |
| YOLOv6-L | 52.8% | 98 fps | 59.6M |

---

## Next Steps

1. **Download val split** (quick test): `python download_coco.py --split val`
2. **Extract**: `python extract_coco.py`
3. **Run evaluation**: `python tools/eval.py --data data/coco.yaml --weights weights/yolov6s.pt --task val`
4. **Compare results** with benchmarks above
5. **Download full dataset** (train + val + test) for training custom models

