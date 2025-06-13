import glob
import os
import openvino as ov
from os.path import basename
from shutil import copy

models=glob.glob("onnxmodels_raw/*.onnx")
os.makedirs("openvinomodels_raw", exist_ok=True)

for m in models:
    print(m)
    ov.save_model(ov.convert_model(m), "openvinomodels_raw/"+basename(m).replace(".onnx", ".xml"))

positional_encodings=glob.glob("onnxmodels_raw/*.npy")
for pe in positional_encodings:
    print(pe)
    copy(pe, pe.replace("onnxmodels_raw", "openvinomodels_raw"))