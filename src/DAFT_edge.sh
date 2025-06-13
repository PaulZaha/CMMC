#!/bin/bash

VENV_PATH=""

source "${VENV_PATH}/bin/activate"


modalities=("Dermoscopy" "Fundus" "Mammography" "Microscopy" "OCT" "US" "XRay")


for modality in "${modalities[@]}"; do
    python finetune.py \
        -pretrained_checkpoint general_finetuned_nonkirsch.pth \
        -num_epochs 5 \
        -batch_size 48 \
        -device cuda \
        -work_dir "work_dir_kirsch_modalities3D/${modality}/" \
        -resume "work_dir_kirsch_modalities3D/${modality}/medsam_lite_latest.pth" \
        --traincsv "datasplit/subset_kirsch_modalities3D/subset_${modality}.train.csv" \
        --valcsv "datasplit/subset_kirsch_modalities3D/subset_${modality}.val.csv"

    if [ $? -ne 0 ]; then
        echo "Fehler beim Ausführen des Befehls für ${modality}. Skript wird beendet."
        exit 1
    fi
done
