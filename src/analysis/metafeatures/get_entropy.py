import os
import argparse
import numpy as np
import csv
from multiprocessing import Pool
from functools import partial
import random
import pandas as pd
from scipy.stats import entropy

def load_npz(npz_path):

    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"Fehler beim Laden der Datei {npz_path}: {e}")
        return None

    if 'imgs' not in data.files:
        print(f"Datei {npz_path} enthält nicht den erforderlichen Schlüssel 'imgs'.")
        return None

    imgs = data['imgs']


    if imgs.ndim == 3:

        if imgs.shape[0] > 1 and imgs.shape[-1] <= 4:

            num_images = imgs.shape[0]
            return imgs
        else:

            imgs = np.expand_dims(imgs, axis=0)
            return imgs
    elif imgs.ndim == 4:

        num_images = imgs.shape[0]
        return imgs
    else:
        print(f"Unerwartete Form der Bilder in {npz_path}: {imgs.shape}")
        return None

def compute_shannon_entropy(image):
    if image.size == 0:
        return 0
    if image.ndim > 1:
        pixel_values = image.flatten()
    else:
        pixel_values = image


    histogram, _ = np.histogram(pixel_values, bins=256, range=(0, 255), density=True)


    histogram = histogram[histogram > 0]


    return entropy(histogram, base=2)

def process_npz_file(npz_path):

    imgs = load_npz(npz_path)
    if imgs is None:
        return []

    num_images = imgs.shape[0]
    if num_images == 0:
        print(f"Keine Bilder in der Datei {npz_path} gefunden.")
        return []


    middle_index = num_images
    image = imgs[middle_index]
    entropy_value = compute_shannon_entropy(image)

    print(f"Verarbeite Datei {npz_path}: Bildindex {middle_index}, Entropie {entropy_value:.4f}")

    return [(npz_path, middle_index, entropy_value)]

def traverse_and_process(input_base_dir, output_csv_path, selected_files=None, sample_fraction=1, num_processes=4, seed=None):

    print(f"Starte im Eingangsverzeichnis: {input_base_dir}")
    print(f"CSV-Ausgabedatei: {output_csv_path}")
    print(f"Stichprobenanteil: {sample_fraction * 100}%")


    npz_files = []
    for root, dirs, files in os.walk(input_base_dir):
        for file in files:
            if file.endswith('.npz') and not file.endswith('_entropy.npz'):
                if selected_files is not None:

                    if file in selected_files:
                        npz_path = os.path.join(root, file)
                        npz_files.append(npz_path)
                else:
                    npz_path = os.path.join(root, file)
                    npz_files.append(npz_path)

    total_files = len(npz_files)
    print(f"Gefundene .npz-Dateien nach Filterung: {total_files}")

    if not npz_files:
        print("Keine .npz-Dateien zum Verarbeiten gefunden.")
        return


    sample_size = max(1, int(total_files * sample_fraction))


    if seed is not None:
        random.seed(seed)

    sampled_files = random.sample(npz_files, sample_size)
    print(f"Ausgewählte .npz-Dateien zum Verarbeiten: {len(sampled_files)}")


    with Pool(processes=num_processes) as pool:
        all_results = pool.map(process_npz_file, sampled_files)


    flat_results = [item for sublist in all_results for item in sublist]

    print(f"Gesammelte Ergebnisse: {len(flat_results)} Einträge")


    try:
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(['file', 'middle_image_index', 'entropy'])

            for row in flat_results:
                writer.writerow(row)
        print(f"Ergebnisse erfolgreich in {output_csv_path} gespeichert.")
    except Exception as e:
        print(f"Fehler beim Schreiben der CSV-Datei {output_csv_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Berechnet die Shannon-Entropie für den mittleren Slice ausgewählter .npz-Dateien und speichert die Ergebnisse in einer CSV-Datei.")
    parser.add_argument('--input', type=str, required=True, help="Pfad zum Eingangs-Basisverzeichnis mit .npz-Dateien.")
    parser.add_argument('--output', type=str, required=True, help="Pfad zur Ausgabedatei (CSV), wo die Ergebnisse gespeichert werden sollen.")
    parser.add_argument('--processes', type=int, default=4, help="Anzahl der parallelen Prozesse (Standard: 4).")
    parser.add_argument('--xlsx', type=str, required=True, help="Pfad zur Excel-Datei (.xlsx), die die zu verarbeitenden Dateinamen in der Spalte 'file' enthält.")
    parser.add_argument('--sample_fraction', type=float, default=1, help="Stichprobenanteil der ausgewählten .npz-Dateien (z.B. 0.05 für 5%). Standard: 0.05")
    parser.add_argument('--seed', type=int, default=None, help="Optionaler Seed für die Zufallsauswahl, um Reproduzierbarkeit zu gewährleisten.")

    args = parser.parse_args()

    input_base_dir = os.path.abspath(args.input)
    output_csv_path = os.path.abspath(args.output)
    num_processes = args.processes
    xlsx_path = os.path.abspath(args.xlsx)
    sample_fraction = args.sample_fraction
    seed = args.seed


    if not os.path.exists(input_base_dir):
        print(f"Eingangsverzeichnis existiert nicht: {input_base_dir}")
        return


    if not os.path.isfile(xlsx_path):
        print(f"Excel-Datei existiert nicht: {xlsx_path}")
        return

    try:
        df = pd.read_excel(xlsx_path)
        if 'file' not in df.columns:
            print(f"Die Excel-Datei {xlsx_path} enthält keine Spalte 'file'.")
            return
        selected_files = df['file'].astype(str).tolist()
    except Exception as e:
        print(f"Fehler beim Lesen der Excel-Datei {xlsx_path}: {e}")
        return


    selected_files = [os.path.basename(f) for f in selected_files]


    if sample_fraction is not None and 0 < sample_fraction < 1:
        if seed is not None:
            random.seed(seed)
        sample_size = max(1, int(len(selected_files) * sample_fraction))
        selected_files = random.sample(selected_files, sample_size)
        print(f"Stichprobe von {sample_size} Dateien aus der Excel-Liste ausgewählt.")

    traverse_and_process(input_base_dir, output_csv_path, selected_files=selected_files, sample_fraction=1.0, num_processes=num_processes, seed=seed)

if __name__ == "__main__":
    main()
