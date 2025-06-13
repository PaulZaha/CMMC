This is the source code repository for the Paper "Do Edges Matter? Investigating Edge-Enhanced Pre-Training for Medical Image Segmentation" part of the DEMI Workshop on the MICCAI Conference 2025.

Environment Details:
WSL Debian
Python 3.11.2
CUDA 12.8 on a Nvidia RTX 4090 (Laptop)
PyTorch 2.4.1

Please be aware to adjust paths in the scripts yourself. Placeholders are included.

Step 1: Copy data in /src/data/raw
For every image modality, a folder needs to be present in the raw directory, containing the modality data.

Step 2: Apply kirsch kernels
python src/kerneling/apply_kernels.py --input inputpath/<MODALITY> --output outputpath/<MODALITY>

Step 2: Download LiteMedSAM weights and put them in `src/work_dir/LiteMedSAM/`, also download [the EfficientViT-SAM l0](https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l0.pt) checkpoint, rename it to l0.pt, put in in `src`

Step 3: Split dataset
python src/split_dataset.py path

Step 4: Create modality datasets
python src/datasplit/modalities3D.py train.csv val.csv
Also do for the edge-enhanced data.

Step 5: Distillation
python src/distill.py -num_epochs 5 -batch_size 8 -device cuda -work_dir work_dir_distill/ -resume work_dir_distill/medsam_lite_latest.pth -pretrained_checkpoint l0.pt --traincsv datasplit/modalities3D/train.csv --valcsv datasplit/modalities3D/val.csv
python modelmerge.py work_dir_distill/medsam_lite_best.pth distilled.pth

Step 6: Pre-training
python src/pretrain.py -pretrained_checkpoint distilled.pth -num_epochs 5 -batch_size 48 -device cuda -work_dir work_dir_general_nonkirsch -resume work_dir_general_nonkirsch/medsam_lite_latest.pth --traincsv datasplit/train.csv --valcsv datasplit/val.csv
python src/pretrain.py -pretrained_checkpoint distilled.pth -num_epochs 5 -batch_size 48 -device cuda -work_dir work_dir_general_kirsch -resume work_dir_general_kirsch/medsam_lite_latest.pth --traincsv datasplit/kirsch_train.csv --valcsv datasplit/kirsch_val.csv

Step 7: Extract weights
python extract_evit.py work_dir_general/medsam_lite_best.pth general_finetuned.pth
python extract_evit.py work_dir_general_kirsch/medsam_lite_best.pth general_finetuned_kirsch.pth
Also for all modalities.

Step 8: Finetuning
mkdir models_edge && ./DAFT_edge.sh
mkdir models_raw && ./DAFT_raw.sh

Step 9: Export onnx
python export_onnx_raw.py
python export_onnx_edge.py
Rename one of the prompt encoders to `prompt_encoder.onnx` and delete all others. They are all shared.

Step 10: Convert to openvino
python onnx2openvino_raw.py
python onnx2openvino_edge.py

Step 11: Inference
Scripts are inference_edge.sh and inference_raw.sh


Scripts for further analysis are in /src/analysis, including the discrete optimization meta-classifier.
