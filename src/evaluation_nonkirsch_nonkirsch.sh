#!/bin/bash

VENV_PATH="/mnt/c/Users/paulz/Documents/DAFTxKerneling/environment"

source "${VENV_PATH}/bin/activate"

# Definiere die Liste der Modalitäten

#modalities=("Dermoscopy")
modalities=("Dermoscopy" "Fundus" "Mammography" "Microscopy" "OCT" "US" "XRay")
# modalities=("US" "XRay")

# Schleife über jede Modalität und führe den Befehl aus
for modality in "${modalities[@]}"; do
    python evaluation/compute_metrics.py \
        -s test_demo/v2_bigval/raw_val/${modality} \
        -g test_demo/v2_bigval/raw_val/${modality} \
        -csv_dir ./results/v3/metrics_raw_${modality}.csv
        

    if [ $? -ne 0 ]; then
        echo "Fehler beim Ausführen des Befehls für ${modality}. Skript wird beendet."
        exit 1
    fi
done
