import onnx
model = onnx.load("weights/yolov6n.onnx")
opset_version = model.opset_import[0].version
print(f"ONNX model opset version: {opset_version}")