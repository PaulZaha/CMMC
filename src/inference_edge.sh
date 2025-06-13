#!/bin/bash

VENV_PATH=""

source "${VENV_PATH}/bin/activate"

# Definiere die Liste der Modalitäten

modalities=("Dermoscopy" "Fundus" "Mammography" "Microscopy" "OCT" "US" "XRay")


for modality in "${modalities[@]}"; do
    python PerfectMetaOpenVINO_edge_val.py \
        --input_csv PATH/datasplit/modalities3D/${modality}.val.csv \
        --output_dir PATH/${modality} \
        

    if [ $? -ne 0 ]; then
        echo "Fehler beim Ausführen des Befehls für ${modality}. Skript wird beendet."
        exit 1
    fi
done
