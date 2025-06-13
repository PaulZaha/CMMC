import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import os

def load_npz(npz_path):

    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"Fehler beim Laden der Datei {npz_path}: {e}")
        return None, None

    if 'imgs' not in data.files or 'gts' not in data.files:
        print(f"Datei {npz_path} enthält nicht die erforderlichen Schlüssel 'imgs' und 'gts'.")
        return None, None

    imgs = data['imgs']
    gts = data['gts']


    if gts.ndim == 3:
        num_masks = gts.shape[0]
    elif gts.ndim == 2:
        num_masks = 1
        gts = np.expand_dims(gts, axis=0) 
    else:
        print(f"Unerwartete Form der Masken in {npz_path}: {gts.shape}")
        return None, None


    if imgs.ndim == 3:

        if imgs.shape[0] == num_masks:

            num_images = imgs.shape[0]
            return imgs, gts
        else:

            num_images = 1

            if imgs.shape[-1] > 4:
                print(f"Warnung: Bild scheint mehrere Kanäle zu haben, aber Anzahl der Masken stimmt nicht überein.")
                return None, None
            imgs = np.expand_dims(imgs, axis=0)
            return imgs, gts
    elif imgs.ndim == 4:

        num_images = imgs.shape[0]
        if num_images == num_masks:
            return imgs, gts
        else:
            print(f"Anzahl der Bilder ({num_images}) stimmt nicht mit der Anzahl der Masken ({num_masks}) in {npz_path} überein.")
            return None, None
    else:
        print(f"Unerwartete Form der Bilder in {npz_path}: {imgs.shape}")
        return None, None

def verarbeite_npz(npz_path):

    imgs, gts = load_npz(npz_path)
    if imgs is None:
        return []


    if imgs.ndim == 4:

        N = imgs.shape[0]
        if N == 0:
            print(f"Keine Bilder in Datei {npz_path} gefunden.")
            return []
        central_idx = N // 2
        central_img = imgs[central_idx, :, :, :]
        brightness = central_img.mean()
        return [(npz_path, central_idx, brightness)]
    elif imgs.ndim == 3:

        N = imgs.shape[0]
        if N == 0:
            print(f"Keine Bilder in Datei {npz_path} gefunden.")
            return []
        central_idx = N // 2
        central_img = imgs[central_idx, :, :]
        brightness = central_img.mean()
        return [(npz_path, central_idx, brightness)]
    else:
        print(f"Unerwartete Bilddimensionalität: {imgs.ndim} in Datei {npz_path}")
        return []

def verarbeite_npz_variance(npz_path):

    imgs, gts = load_npz(npz_path)
    if imgs is None:
        return []
    if imgs.ndim == 4:

        N = imgs.shape[0]
        if N == 0:
            print(f"Keine Bilder in Datei {npz_path} gefunden.")
            return []
        central_idx = N // 2
        central_img = imgs[central_idx, :, :, :]
        var = np.var(central_img)
        return [(npz_path, central_idx, var)]
    elif imgs.ndim == 3:
        N = imgs.shape[0]
        if N == 0:
            print(f"Keine Bilder in Datei {npz_path} gefunden.")
            return []
        central_idx = N // 2
        central_img = imgs[central_idx, :, :]
        var = np.var(central_img)
        return [(npz_path, central_idx, var)]
    else:
        print(f"Unerwartete Bilddimensionalität: {imgs.ndim} in Datei {npz_path}")
        return []



