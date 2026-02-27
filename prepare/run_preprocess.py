import os
from ast import literal_eval
from glob import glob
import math

import numpy as np
import pandas as pd
import SimpleITK as sitk

from configs import (AUTO_ADD_NEGATIVE_CENTER_FROM_IMAGE,
                     AUTO_DISCOVER_DEFAULT_CLASS, AUTO_DISCOVER_MASK_SUFFIX,
                     AUTO_DISCOVER_NIFTI_GZ, AUTO_GENERATE_MANIFEST_FROM_DIRS,
                     IMAGE_DIR, KFOLD_SEED, KFOLD_SPLITS, LABEL_DIR,
                     MANIFEST_IMAGE_PATH_RELATIVE_TO_RESOURCES, MANIFEST_OUTPUT_DIR,
                     MANIFEST_PATH, MASK_PATH_RELATIVE_TO_RESOURCES,
                     NIFTI_GLOB_PATTERN, OUTPUT_PATH, RESOURCES_PATH)
from prepare._classes import CTScan
from prepare.manifest_builder import generate_kfold_manifests


LEGACY_META_COLUMNS = ['seriesuid', 'spacing', 'lungs_bounding_box', 'centers', 'radii', 'class']
REQUIRED_MANIFEST_COLUMNS = ['case_id', 'image_path']


def _load_legacy_annotations():
    annotations = pd.read_csv(RESOURCES_PATH + '/annotations.csv')
    candidates = pd.read_csv(RESOURCES_PATH + '/candidates.csv')
    return annotations, candidates


def _get_positive_series(annotations):
    paths = glob(RESOURCES_PATH + '/*/' + "*.mhd")
    file_list = [f.split('/')[-1][:-4] for f in paths]
    series = annotations['seriesuid'].tolist()
    infected = [f for f in file_list if f in series]
    return infected


def _get_negative_series(annotations):
    paths = glob(RESOURCES_PATH + '/*/' + "*.mhd")
    file_list = [f.split('/')[-1][:-4] for f in paths]
    series = annotations['seriesuid'].tolist()
    cleans = [f for f in file_list if f not in series]
    return cleans


def _ensure_output_dirs():
    [os.makedirs(d, exist_ok=True) for d in
     [f'{OUTPUT_PATH}/preprocessed/positives', f'{OUTPUT_PATH}/preprocessed/negatives']]


def _to_python_list(value):
    if isinstance(value, str):
        return literal_eval(value)
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return list(value)


def _load_manifest():
    manifest_path = MANIFEST_PATH
    if manifest_path is None:
        return None
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(RESOURCES_PATH, manifest_path)
    manifest = pd.read_csv(manifest_path)
    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f'Manifest missing required columns: {missing}')

    has_label_columns = all(c in manifest.columns for c in ('class', 'centers', 'radii'))
    has_mask_column = 'mask_path' in manifest.columns
    if not has_label_columns and not has_mask_column:
        raise ValueError(
            'Manifest must contain either (class, centers, radii) columns or a mask_path column.'
        )
    return manifest




def _autodiscover_nifti_manifest():
    # >>> NII_GZ_AUTO_START: optional auto-discovery without a user-provided manifest
    discovered_paths = glob(os.path.join(RESOURCES_PATH, NIFTI_GLOB_PATTERN), recursive=True)
    discovered_paths = sorted([p for p in discovered_paths if p.endswith('.nii.gz')])
    if len(discovered_paths) == 0:
        return None

    rows = []
    for image_path in discovered_paths:
        case_id = os.path.basename(image_path).replace('.nii.gz', '')
        row = {
            'case_id': case_id,
            'image_path': image_path,
            'class': AUTO_DISCOVER_DEFAULT_CLASS,
            'centers': '[]',
            'radii': '[]',
        }
        if AUTO_DISCOVER_MASK_SUFFIX is not None and image_path.endswith('.nii.gz'):
            candidate_mask_path = image_path[:-7] + AUTO_DISCOVER_MASK_SUFFIX
            if os.path.exists(candidate_mask_path):
                row['mask_path'] = candidate_mask_path
        rows.append({
            **row,
        })
    return pd.DataFrame(rows)
    # >>> NII_GZ_AUTO_END

def _resolve_image_path(image_path):
    if MANIFEST_IMAGE_PATH_RELATIVE_TO_RESOURCES and not os.path.isabs(image_path):
        return os.path.join(RESOURCES_PATH, image_path)
    return image_path


def _resolve_mask_path(mask_path):
    if MASK_PATH_RELATIVE_TO_RESOURCES and not os.path.isabs(mask_path):
        return os.path.join(RESOURCES_PATH, mask_path)
    return mask_path


def _derive_targets_from_instance_mask(image_path, mask_path):
    ds = sitk.ReadImage(image_path)
    spacing = np.array(list(reversed(ds.GetSpacing())), dtype=np.float32)
    origin = np.array(list(reversed(ds.GetOrigin())), dtype=np.float32)

    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.int32)
    instance_ids = [int(i) for i in np.unique(mask) if int(i) > 0]

    centers = []
    radii = []
    voxel_volume = float(np.prod(spacing))

    for instance_id in instance_ids:
        voxels = np.argwhere(mask == instance_id)
        if voxels.shape[0] == 0:
            continue
        centroid_voxel = voxels.mean(axis=0)
        world_center = origin + centroid_voxel * spacing
        centers.append(tuple(world_center.tolist()))

        volume_mm3 = float(voxels.shape[0]) * voxel_volume
        radius_mm = ((3.0 * volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0)
        radii.append(radius_mm)

    clazz = 1 if len(centers) > 0 else 0
    return clazz, centers, radii


def _add_synthetic_negative_center_if_needed(image_path, clazz, centers, radii):
    if clazz != 0 or len(centers) > 0 or not AUTO_ADD_NEGATIVE_CENTER_FROM_IMAGE:
        return centers, radii
    ds = sitk.ReadImage(image_path)
    spacing = np.array(list(reversed(ds.GetSpacing())), dtype=np.float32)
    origin = np.array(list(reversed(ds.GetOrigin())), dtype=np.float32)
    image = sitk.GetArrayFromImage(ds)
    center_voxel = np.array(image.shape, dtype=np.float32) / 2.0
    world_center = origin + center_voxel * spacing
    return [tuple(world_center.tolist())], [1.0]


def _save_preprocessed_from_manifest(manifest_df):
    _ensure_output_dirs()
    meta_data = pd.DataFrame(columns=LEGACY_META_COLUMNS + ['image_path'])
    for rec in manifest_df.to_dict(orient='records'):
        series_id = str(rec['case_id'])
        image_path = _resolve_image_path(rec['image_path'])

        mask_path = rec.get('mask_path', None)
        if isinstance(mask_path, str) and len(mask_path) > 0:
            mask_path = _resolve_mask_path(mask_path)
            clazz, centers, radii = _derive_targets_from_instance_mask(image_path=image_path, mask_path=mask_path)
        else:
            clazz = int(rec.get('class', AUTO_DISCOVER_DEFAULT_CLASS))
            centers = _to_python_list(rec.get('centers', []))
            radii = _to_python_list(rec.get('radii', []))

        centers, radii = _add_synthetic_negative_center_if_needed(
            image_path=image_path, clazz=clazz, centers=centers, radii=radii
        )

        extra_meta = {
            k: rec[k] for k in rec.keys()
            if k not in {'case_id', 'image_path', 'mask_path', 'class', 'centers', 'radii'}
        }

        ct = CTScan(
            seriesuid=series_id,
            centers=centers,
            radii=radii,
            clazz=clazz,
            image_path=image_path,
            extra_meta=extra_meta,
        )
        ct.preprocess()
        ct.save_preprocessed_image()
        diction = ct.get_info_dict()
        meta_data = meta_data.append(pd.Series(diction), ignore_index=True)
    meta_data.to_csv(f'{OUTPUT_PATH}/preprocessed_meta.csv')


def _save_preprocessed_legacy():
    _ensure_output_dirs()
    annotations, candidates = _load_legacy_annotations()
    meta_data = pd.DataFrame(columns=LEGACY_META_COLUMNS)
    for series_id in _get_positive_series(annotations):
        nodule_coords_annot = annotations[annotations['seriesuid'] == series_id]
        tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords_annot.iloc]
        radii = [(a['diameter_mm'] / 2) for a in nodule_coords_annot.iloc]
        ct = CTScan(seriesuid=series_id, centers=tp_co, radii=radii, clazz=1)
        ct.preprocess()
        ct.save_preprocessed_image()
        diction = ct.get_info_dict()
        meta_data = meta_data.append(pd.Series(diction), ignore_index=True)
    for series_id in _get_negative_series(annotations):
        nodule_coords_candid = candidates[candidates['seriesuid'] == series_id]
        tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords_candid.iloc]
        radii = list(np.random.randint(40, size=len(tp_co)))
        max_numbers_to_use = min(len(tp_co), 3)
        tp_co = tp_co[:max_numbers_to_use]
        radii = radii[:max_numbers_to_use]
        ct = CTScan(seriesuid=series_id, centers=tp_co, radii=radii, clazz=0)
        ct.preprocess()
        ct.save_preprocessed_image()
        diction = ct.get_info_dict()
        meta_data = meta_data.append(pd.Series(diction), ignore_index=True)
    meta_data.to_csv(f'{OUTPUT_PATH}/preprocessed_meta.csv')


def _maybe_generate_manifest_from_dirs():
    if not AUTO_GENERATE_MANIFEST_FROM_DIRS:
        return None
    if IMAGE_DIR is None or LABEL_DIR is None:
        raise ValueError('AUTO_GENERATE_MANIFEST_FROM_DIRS=True requires IMAGE_DIR and LABEL_DIR.')

    out_dir = MANIFEST_OUTPUT_DIR or os.path.join(OUTPUT_PATH, 'manifests')
    result = generate_kfold_manifests(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        output_dir=out_dir,
        n_splits=KFOLD_SPLITS,
        seed=KFOLD_SEED,
    )
    print(f"Generated {result['num_cases']} matched image/mask cases into: {out_dir}")
    print(f"All-cases manifest: {result['all_manifest_path']}")
    print(f"Fold summary: {result['summary_path']}")

    # default preprocessing manifest: all cases
    return pd.read_csv(result['all_manifest_path'])


def save_preprocessed_data():
    manifest = _load_manifest()

    if manifest is None:
        manifest = _maybe_generate_manifest_from_dirs()

    if manifest is None and AUTO_DISCOVER_NIFTI_GZ:
        manifest = _autodiscover_nifti_manifest()
        if manifest is not None:
            print(f'Running in auto-discovery NIfTI mode with {len(manifest)} scans.')
            _save_preprocessed_from_manifest(manifest)
            return

    if manifest is None:
        print('Running in legacy LUNA16 mode (mhd + annotations/candidates CSVs).')
        _save_preprocessed_legacy()
    else:
        print(f'Running in manifest mode with {len(manifest)} scans.')
        _save_preprocessed_from_manifest(manifest)


if __name__ == '__main__':
    save_preprocessed_data()
