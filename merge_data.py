#!/usr/bin/env python3
"""
merge_chunks.py — Fusionne tous les chunks tokens.npy en un seul fichier
Usage : python merge_chunks.py
"""

import os
import sys
import time
import numpy as np

DATA_DIR = './data_exp'
OUT_FILE = './data_exp/chunk_000/tokens.npy'

arrays = []
for entry in sorted(os.listdir(DATA_DIR)):
    chunk_dir = os.path.join(DATA_DIR, entry)
    if not os.path.isdir(chunk_dir) or not entry.startswith('chunk'):
        continue
    for name in ('tokens.npy', 'cosmopedia.npy'):
        npy = os.path.join(chunk_dir, name)
        if os.path.exists(npy):
            arr = np.load(npy)
            print(f'  {entry} : {len(arr) / 1e9:.3f}B tokens  dtype={arr.dtype}')
            arrays.append(arr)
            break

if not arrays:
    print('ERREUR : aucun chunk trouvé')
    sys.exit(1)

print(f'\n  Fusion de {len(arrays)} chunks...')
t0     = time.time()
merged = np.concatenate(arrays)
del arrays

print(f'  Total : {len(merged) / 1e9:.3f}B tokens  dtype={merged.dtype}')
print(f'  Sauvegarde → {OUT_FILE}')
np.save(OUT_FILE, merged)
del merged

print(f'  Terminé en {time.time() - t0:.1f}s')
print(f'  Vérification...')
check = np.load(OUT_FILE, mmap_mode='r')
print(f'  OK : {len(check) / 1e9:.3f}B tokens  dtype={check.dtype}')
