import os
import numpy as np
from scipy.ndimage import convolve
import argparse
from multiprocessing import Pool
from functools import partial

def apply_kirsch_edge_detection(image):
    kirsch_kernels = [
        np.array([[5, 5, 5],
                  [-3, 0, -3],
                  [-3, -3, -3]]),
        np.array([[5, 5, -3],
                  [5, 0, -3],
                  [-3, -3, -3]]),
        np.array([[5, -3, -3],
                  [5, 0, -3],
                  [5, -3, -3]]),
        np.array([[-3, -3, -3],
                  [5, 0, -3],
                  [5, 5, -3]]),
        np.array([[-3, -3, -3],
                  [-3, 0, -3],
                  [5, 5, 5]]),
        np.array([[-3, -3, -3],
                  [-3, 0, 5],
                  [-3, 5, 5]]),
        np.array([[-3, -3, 5],
                  [-3, 0, 5],
                  [-3, -3, 5]]),
        np.array([[-3, 5, 5],
                  [-3, 0, 5],
                  [-3, -3, -3]])
    ]


    edge_magnitude = np.zeros_like(image, dtype=float)


    for kernel in kirsch_kernels:
        response = convolve(image.astype(float), kernel, mode='constant', cval=0.0)
        edge_magnitude = np.maximum(edge_magnitude, response)


    edge_magnitude -= edge_magnitude.min()
    if edge_magnitude.max() != 0:
        edge_magnitude /= edge_magnitude.max()
    edge_magnitude *= 255.0
    edge_image = edge_magnitude.astype(np.uint8)

    return edge_image

def process_npz_file(npz_path, output_npz_path):

    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"Fehler beim Laden der Datei {npz_path}: {e}")
        return


    image_key = ['imgs']


    # Holen des Bildarrays
    image_array = data[image_key]
    print(f"Verarbeite {npz_path}: Array-Form {image_array.shape}")


    slice_axis = 0



    permute_order = [slice_axis] + [i for i in range(image_array.ndim) if i != slice_axis]
    image_array_permuted = np.transpose(image_array, permute_order)


    num_slices = image_array_permuted.shape[0]


    edge_array = np.zeros_like(image_array_permuted, dtype=np.uint8)


    for i in range(num_slices):
        edge_array[i] = apply_kirsch_edge_detection(image_array_permuted[i])


        print(f"  Verarbeitet Slice {i + 1} von {num_slices}")


    inverse_permute_order = np.argsort(permute_order)
    edge_array_original_order = np.transpose(edge_array, inverse_permute_order)


    new_data = {}
    for key in data.files:
        if key == image_key:
            new_data[key] = edge_array_original_order
        else:
            new_data[key] = data[key]


    try:
        np.savez_compressed(output_npz_path, **new_data)
        print(f"Gespeichert: {output_npz_path}")
    except Exception as e:
        print(f"Fehler beim Speichern der Datei {output_npz_path}: {e}")


def traverse_and_process(input_base_dir, output_base_dir):
    for root, dirs, files in os.walk(input_base_dir):

        npz_files = [f for f in files if f.endswith('.npz')]
        if npz_files:
            for npz_file in npz_files:
                input_npz_path = os.path.join(root, npz_file)


                relative_path = os.path.relpath(root, input_base_dir)


                output_dir = os.path.join(output_base_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)


                output_npz_path = os.path.join(output_dir, npz_file)


                process_npz_file(input_npz_path, output_npz_path)
        else:

            pass

def process_subdir(subdir, input_base_dir, output_base_dir):

    if subdir == '.':
        input_subdir = input_base_dir
        output_subdir = output_base_dir
    else:
        input_subdir = os.path.join(input_base_dir, subdir)
        output_subdir = os.path.join(output_base_dir, subdir)
    os.makedirs(output_subdir, exist_ok=True)
    traverse_and_process(input_subdir, output_subdir)
def main():
    parser = argparse.ArgumentParser(description="Wendet Kirsch-Kantenerkennung auf alle Slices in .npz-Dateien an und speichert die Ergebnisse.")
    parser.add_argument('--input', type=str, required=True, help="Pfad zum Eingangs-Basisverzeichnis (train_npz).")
    parser.add_argument('--output', type=str, required=True, help="Pfad zum Ausgabebasisverzeichnis, wo die verarbeiteten .npz-Dateien gespeichert werden sollen.")

    args = parser.parse_args()

    input_base_dir = os.path.abspath(args.input)
    output_base_dir = os.path.abspath(args.output)


    if not os.path.exists(input_base_dir):
        print(f"Eingangsverzeichnis existiert nicht: {input_base_dir}")
        return


    os.makedirs(output_base_dir, exist_ok=True)


    subdirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]

    has_npz_files = any(f.endswith('.npz') for f in os.listdir(input_base_dir))
    if has_npz_files:
        subdirs.append('.')


    pool = Pool(processes=8)
    process_subdir_partial = partial(process_subdir, input_base_dir=input_base_dir, output_base_dir=output_base_dir)
    pool.map(process_subdir_partial, subdirs)
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()