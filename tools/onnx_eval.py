#!/usr/bin/env python3
import argparse
import os
import sys
import json
import numpy as np
import torch
import onnxruntime as ort
import time
from pathlib import Path

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.data.data_load import create_dataloader
from yolov6.utils.events import LOGGER
from yolov6.utils.nms import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='weights/yolov6s.onnx')
    parser.add_argument('--data', type=str, default='data/coco.yaml')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', type=str, default='dml', help="'dml' or 'cpu'")
    parser.add_argument('--anno', type=str, default='', help='Optional path to COCO instances_val2017.json')
    parser.add_argument('--conf-thres', type=float, default=0.03)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--save-dir', type=str, default='runs/val/onnx_dml')
    return parser.parse_args()


def to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # session
    providers = []
    if args.device.lower() == 'dml' and 'DmlExecutionProvider' in ort.get_available_providers():
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(args.onnx, sess_opts, providers=providers)
    input_name = sess.get_inputs()[0].name
    print('Using providers:', sess.get_providers())
    print('ONNX input name:', input_name)

    # dataloader
    # load yaml to get val path via Evaler.reload_dataset logic: reuse simple parse
    import yaml
    with open(args.data, errors='ignore') as f:
        data = yaml.safe_load(f)
    # allow overriding the annotation path via CLI
    if args.anno:
        data['anno_path'] = args.anno
    val_path = data.get('val')
    if not isinstance(val_path, list):
        val_path = [val_path]

    # create dataloader(s)
    # Using first val path and stride 32 (the model stride may vary, but dataloader only needs a stride)
    # We set stride=32 because create_dataloader requires it; the actual images will be letterboxed to img_size.
    dataloader, dataset = create_dataloader(val_path[0], args.img_size, args.batch_size, stride=32, pad=0.0, rect=False, data_dict=data, task='val')

    all_preds = []
    start_time = time.time()
    total_imgs = 0
    for i, (imgs, targets, paths, shapes) in enumerate(dataloader):
        # imgs: torch tensor in shape (N,3,H,W), values 0-255, RGB (per repo dataset)
        imgs_np = imgs.numpy().astype(np.float32) / 255.0  # normalize to 0-1
        # ONNXRuntime expects NCHW float32
        ort_inputs = {input_name: imgs_np}
        t0 = time.time()
        outs = sess.run(None, ort_inputs)
        t1 = time.time()
        # outs[0] expected shape (N, anchors, 5+nc) similar to model output
        out_tensor = torch.from_numpy(np.array(outs[0]))  # convert to torch

        # Run repo NMS (which expects tensor in same format as model export)
        preds = non_max_suppression(out_tensor, args.conf_thres, args.iou_thres, None, False, max_det=args.max_det)

        # convert preds to coco format (similar to Evaler.convert_to_coco_format)
        for b_i, pred in enumerate(preds):
            if pred is None or len(pred) == 0:
                continue
            path = Path(paths[b_i])
            shape = shapes[b_i][0]
            # preds come as xyxy absolute in padded coords, but we follow Evaler's conversion approach
            # rescale coords back to original image
            pred[:, :4] = pred[:, :4].clamp(0)
            # convert to xywh normalized by original image size
            # use Evaler.box_convert to get center/w/h then normalize by gn
            gn = torch.tensor([shape[1], shape[0], shape[1], shape[0]])
            xywh = ((pred[:, :4].clone()).view(-1,4))
            # convert xyxy to xywh
            y = xywh.clone()
            y[:,0] = (xywh[:,0] + xywh[:,2]) / 2
            y[:,1] = (xywh[:,1] + xywh[:,3]) / 2
            y[:,2] = xywh[:,2] - xywh[:,0]
            y[:,3] = xywh[:,3] - xywh[:,1]
            y = (y / gn).tolist()
            cls = pred[:, 5].tolist()
            scores = pred[:,4].tolist()
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            for j, (cbox, ccls, cs) in enumerate(zip(y, cls, scores)):
                bbox = [round(x, 3) for x in cbox]
                pred_data = {
                    "image_id": image_id,
                    "category_id": int(ccls),
                    "bbox": bbox,
                    "score": round(cs, 5)
                }
                all_preds.append(pred_data)
        total_imgs += imgs.shape[0]
        if (i+1) % 10 == 0:
            print(f'Processed batches: {i+1}, images: {total_imgs}, last batch inference time: {t1-t0:.3f}s')

    print('Writing predictions to file...')
    pred_json = os.path.join(args.save_dir, 'predictions.json')
    with open(pred_json, 'w') as f:
        json.dump(all_preds, f)
    print('Done. Time elapsed:', time.time() - start_time)

    # run COCOeval
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        anno = COCO(data.get('anno_path'))
        pred = anno.loadRes(pred_json)
        cocoEval = COCOeval(anno, pred, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    except Exception as e:
        print('COCOeval failed:', e)


if __name__ == '__main__':
    main()
