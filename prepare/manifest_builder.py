import os
from glob import glob
from typing import List

import numpy as np
import pandas as pd


def _ensure_nii_gz(path: str) -> bool:
    return path.endswith('.nii.gz')


def _case_id_from_filename(path: str) -> str:
    return os.path.basename(path).replace('.nii.gz', '')


def _match_image_and_mask_paths(image_dir: str, label_dir: str) -> List[dict]:
    image_paths = sorted([p for p in glob(os.path.join(image_dir, '*.nii.gz')) if _ensure_nii_gz(p)])
    if len(image_paths) == 0:
        raise ValueError(f'No .nii.gz files found in image_dir: {image_dir}')

    rows = []
    missing_masks = []
    for image_path in image_paths:
        fname = os.path.basename(image_path)
        mask_path = os.path.join(label_dir, fname)
        if not os.path.exists(mask_path):
            missing_masks.append(fname)
            continue
        rows.append({
            'case_id': _case_id_from_filename(image_path),
            'image_path': image_path,
            'mask_path': mask_path,
        })

    if len(missing_masks) > 0:
        preview = ', '.join(missing_masks[:10])
        raise ValueError(f'Missing masks for {len(missing_masks)} images, examples: {preview}')

    if len(rows) == 0:
        raise ValueError('No matched image/mask pairs found.')
    return rows


def generate_kfold_manifests(image_dir: str, label_dir: str, output_dir: str, n_splits: int = 5, seed: int = 2026):
    rows = _match_image_and_mask_paths(image_dir=image_dir, label_dir=label_dir)
    df = pd.DataFrame(rows).sort_values('case_id').reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    all_manifest_path = os.path.join(output_dir, 'manifest_all.csv')
    df.to_csv(all_manifest_path, index=False)

    rng = np.random.RandomState(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    fold_paths = []
    for fold_idx in range(n_splits):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(n_splits) if i != fold_idx])

        fold_df = df.copy()
        fold_df['split'] = 'train'
        fold_df.loc[val_idx, 'split'] = 'val'

        fold_manifest_path = os.path.join(output_dir, f'manifest_fold{fold_idx}.csv')
        fold_df.to_csv(fold_manifest_path, index=False)

        train_manifest_path = os.path.join(output_dir, f'manifest_fold{fold_idx}_train.csv')
        val_manifest_path = os.path.join(output_dir, f'manifest_fold{fold_idx}_val.csv')
        fold_df.loc[train_idx].to_csv(train_manifest_path, index=False)
        fold_df.loc[val_idx].to_csv(val_manifest_path, index=False)

        fold_paths.append({
            'fold': fold_idx,
            'fold_manifest': fold_manifest_path,
            'train_manifest': train_manifest_path,
            'val_manifest': val_manifest_path,
            'train_count': int(len(train_idx)),
            'val_count': int(len(val_idx)),
        })

    summary_path = os.path.join(output_dir, 'manifest_summary.csv')
    pd.DataFrame(fold_paths).to_csv(summary_path, index=False)

    return {
        'all_manifest_path': all_manifest_path,
        'summary_path': summary_path,
        'folds': fold_paths,
        'num_cases': int(len(df)),
    }
