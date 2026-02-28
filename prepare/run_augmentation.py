import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from prepare._classes import PatchMaker
import pandas as pd
from configs import AUG_SKIP_EXISTING, OUTPUT_PATH, PREPROCESSED_SAVE_FORMAT
from ast import literal_eval


def _get_patches(record):
    rec = record
    seriesuid = rec['seriesuid']
    spacing = literal_eval(rec['spacing'])
    lungs_bounding_box = literal_eval(rec['lungs_bounding_box'])
    centers = literal_eval(rec['centers'])
    radii = literal_eval(rec['radii'])
    clazz = int(rec['class'])
    file_directory = 'preprocessed/positives' if clazz == 1 else 'preprocessed/negatives'
    # >>> NII_GZ_AUTO_START: read preprocessed inputs saved as npy/npz
    extension = PREPROCESSED_SAVE_FORMAT.lower()
    file_path = f'{OUTPUT_PATH}/{file_directory}/{seriesuid}.{extension}'
    # >>> NII_GZ_AUTO_END
    pm = PatchMaker(seriesuid=seriesuid, coords=centers, radii=radii, spacing=spacing,
                    lungs_bounding_box=lungs_bounding_box,
                    file_path=file_path, clazz=clazz)
    return pm.get_augmented_patches()


def save_augmented_data(preprocess_meta):
    [os.makedirs(d, exist_ok=True) for d in
     [f'{OUTPUT_PATH}/augmented/positives', f'{OUTPUT_PATH}/augmented/negatives']]
    existing_meta_path = f'{OUTPUT_PATH}/augmented_meta.csv'
    if AUG_SKIP_EXISTING and os.path.exists(existing_meta_path):
        augmentation_meta = pd.read_csv(existing_meta_path, index_col=0)
    else:
        augmentation_meta = pd.DataFrame(columns=['seriesuid', 'sub_index', 'centers', 'lungs_bounding_box', 'radii',
                                                  'class'])
    list_of_positives = []
    list_of_negatives = []
    for rec in preprocess_meta.loc[preprocess_meta['class'] == 1].iloc:
        list_of_positives += _get_patches(rec)
    for rec in preprocess_meta.loc[preprocess_meta['class'] == 0].iloc:
        list_of_negatives += _get_patches(rec)
        # 33 percent of the data will be negative samples
        if len(list_of_negatives) > len(list_of_positives) / 2:
            break
    new_meta = pd.DataFrame(list_of_positives + list_of_negatives)
    if not new_meta.empty:
        augmentation_meta = pd.concat([augmentation_meta, new_meta], ignore_index=True)
        augmentation_meta = augmentation_meta.drop_duplicates(subset=['seriesuid', 'sub_index', 'class'], keep='first')
    augmentation_meta.to_csv(existing_meta_path)


if __name__ == '__main__':
    p_meta = pd.read_csv(f'{OUTPUT_PATH}/preprocessed_meta.csv', index_col=0)
    save_augmented_data(p_meta)
