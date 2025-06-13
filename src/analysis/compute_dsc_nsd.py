import numpy as np
import os
from os import listdir
from os.path import join, basename
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import multiprocessing as mp
import argparse

epsilon = 1e-7  # kleiner Wert zum Stabilisieren der Log-Berechnung

def compute_binary_dsc(gt, seg):
    intersection = np.sum((gt > 0) & (seg > 0))
    union = np.sum(gt > 0) + np.sum(seg > 0)
    if union == 0:
        return 1.0  # Beide sind leer
    return (2.0 * intersection) / union

def compute_binary_nsd(gt, seg, spacing, tolerance=2.0):
    if gt.ndim == 2:
        gt = gt[np.newaxis, :, :]
        seg = seg[np.newaxis, :, :]
    elif gt.ndim != 3:
        raise ValueError("gt und seg müssen entweder 2D oder 3D Arrays sein.")
    
    if np.sum(gt) == 0 and np.sum(seg) == 0:
        return 1.0
    elif np.sum(gt) == 0 or np.sum(seg) == 0:
        return 0.0
    else:
        try:
            surface_distance = compute_surface_distances(gt, seg, spacing_mm=spacing)
            return compute_surface_dice_at_tolerance(surface_distance, tolerance)
        except Exception as e:
            print(f"Fehler bei der Berechnung der NSD : {e}")
            return 0.0

def compute_binary_crossentropy_loss(gt, seg):
    """
    Berechnet den Binary Cross-Entropy Loss.
    Annahme: gt und seg sind binäre Masken (0 oder 1).
    """
    # Konvertiere zu float und clippe, um log(0) zu vermeiden
    seg = seg.astype(np.float32)
    seg = np.clip(seg, epsilon, 1.0 - epsilon)
    loss = - (gt * np.log(seg) + (1 - gt) * np.log(1 - seg))
    return np.mean(loss)

def compute_dice_loss(gt, seg):
    """
    Dice Loss = 1 - Dice Coefficient
    """
    dsc = compute_binary_dsc(gt, seg)
    return 1 - dsc

def compute_iou_loss(gt, seg):
    """
    Intersection over Union Loss = 1 - IoU
    """
    intersection = np.sum((gt > 0) & (seg > 0))
    union = np.sum(gt > 0) + np.sum(seg > 0) - intersection
    # Falls beide Masken leer sind, wird IoU als 1 definiert
    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union
    return 1 - iou

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seg_dir', default='test_demo/segs', type=str)
parser.add_argument('-g', '--gt_dir', default='test_demo/gts', type=str)
parser.add_argument('-csv_dir', default='test_demo/metrics.csv', type=str)
parser.add_argument('-num_workers', type=int, default=16)
parser.add_argument('-nsd', default=True, type=bool, help='set it to False to disable NSD computation and save time')
args = parser.parse_args()

seg_dir = args.seg_dir
gt_dir = args.gt_dir
csv_dir = args.csv_dir
num_workers = args.num_workers
compute_NSD = args.nsd

def compute_metrics(npz_name):
    dsc = -1.0
    nsd = -1.0
    bce_loss = -1.0
    dice_loss = -1.0
    iou_loss = -1.0

    try:
        npz_seg = np.load(join(seg_dir, npz_name), allow_pickle=True, mmap_mode='r')
        npz_gt = np.load(join(gt_dir, npz_name), allow_pickle=True, mmap_mode='r')
    except Exception as e:
        print(f"Fehler beim Laden der Datei {npz_name}: {e}")
        return npz_name, dsc, nsd, bce_loss, dice_loss, iou_loss

    if 'gts' not in npz_gt or 'segs' not in npz_seg:
        print(f"Datei {npz_name} enthält nicht die erforderlichen Schlüssel 'gts' und 'segs'.")
        return npz_name, dsc, nsd, bce_loss, dice_loss, iou_loss

    gts = npz_gt['gts']
    segs = npz_seg['segs']
    
    # Binarisieren der Ground Truths (alle nicht-null Klassen als 1)
    gts_binary = (gts > 0).astype(np.uint8)
    segs_binary = (segs > 0).astype(np.uint8)
    
    # Debugging-Ausgaben
    print(f"Processing {npz_name}:")
    print(f"gts_binary unique values: {np.unique(gts_binary)}")
    print(f"segs_binary unique values: {np.unique(segs_binary)}")
    print(f"gts_binary sum: {np.sum(gts_binary)}, segs_binary sum: {np.sum(segs_binary)}")
    
    if npz_name.startswith('3D'):
        if 'spacing' in npz_gt:
            spacing = npz_gt['spacing']
        else:
            print(f"Datei {npz_name} ist als 3D markiert, enthält aber kein 'spacing'.")
            spacing = [1.0, 1.0, 1.0]
    else:
        spacing = [1.0, 1.0, 1.0]
    
    # Berechnung der binären DSC
    #dsc = compute_binary_dsc(gts_binary, segs_binary)
    
    # Berechnung der binären NSD
    # if compute_NSD:
    #     if dsc > 0.2:
    #         nsd = compute_binary_nsd(gts_binary, segs_binary, spacing)
    #     else:
    #         nsd = 0.0

    # Berechnung der zusätzlichen Losses
    bce_loss = compute_binary_crossentropy_loss(gts_binary, segs_binary)
    dice_loss = compute_dice_loss(gts_binary, segs_binary)
    iou_loss = compute_iou_loss(gts_binary, segs_binary)
    
    print(f"dsc: {dsc}, nsd: {nsd}, bce_loss: {bce_loss}, dice_loss: {dice_loss}, iou_loss: {iou_loss}")
    return npz_name, dsc, nsd, bce_loss, dice_loss, iou_loss

if __name__ == '__main__':
    seg_metrics = OrderedDict()
    seg_metrics['case'] = []
    seg_metrics['dsc'] = []
    if compute_NSD:
        seg_metrics['nsd'] = []
    seg_metrics['bce_loss'] = []
    seg_metrics['dice_loss'] = []
    seg_metrics['iou_loss'] = []
    
    npz_names = [f for f in os.listdir(gt_dir) if f.endswith('.npz')]
    
    batch_size = 10  # Größe des Batches
    batch_count = 0  # Zähler für die Batches
    
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(npz_names)) as pbar:
            for npz_name, dsc, nsd, bce_loss, dice_loss, iou_loss in pool.imap_unordered(compute_metrics, npz_names):
                seg_metrics['case'].append(npz_name)
                seg_metrics['dsc'].append(np.round(dsc, 4))
                if compute_NSD:
                    seg_metrics['nsd'].append(np.round(nsd, 4))
                seg_metrics['bce_loss'].append(np.round(bce_loss, 4))
                seg_metrics['dice_loss'].append(np.round(dice_loss, 4))
                seg_metrics['iou_loss'].append(np.round(iou_loss, 4))
                
                # Speichern nach jedem Batch
                if len(seg_metrics['case']) >= batch_size:
                    batch_df = pd.DataFrame(seg_metrics)
                    batch_df = batch_df.sort_values(by=['case'])
                    batch_df.to_csv(csv_dir, mode='a', header=not os.path.exists(csv_dir), index=False)
                    
                    # Reset der Ergebnisse nach jedem Batch
                    seg_metrics['case'] = []
                    seg_metrics['dsc'] = []
                    if compute_NSD:
                        seg_metrics['nsd'] = []
                    seg_metrics['bce_loss'] = []
                    seg_metrics['dice_loss'] = []
                    seg_metrics['iou_loss'] = []
                    
                    batch_count += 1
                pbar.update()
    
    # Finales Speichern der verbleibenden Daten, falls weniger als batch_size übrig sind
    if len(seg_metrics['case']) > 0:
        final_df = pd.DataFrame(seg_metrics)
        final_df = final_df.sort_values(by=['case'])
        final_df.to_csv(csv_dir, mode='a', header=not os.path.exists(csv_dir), index=False)
