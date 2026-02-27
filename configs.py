# Directory used to save intermediate files, checkpoints, and metadata.
OUTPUT_PATH = '/home/ai/Luna16-master/output'

# Dataset root used in legacy LUNA16 mode (mhd/raw + annotations.csv/candidates.csv).
RESOURCES_PATH = '/Users/mostafa/Desktop/dsb_analyse/input'


# -------- Industrial dataset options --------
# When MANIFEST_PATH is provided, preprocessing reads this file instead of relying on
# LUNA16 annotations/candidates conventions.
#
# Required columns:
#   - case_id: unique case identifier
#   - image_path: absolute or RESOURCES_PATH-relative path to the image (supports .nii.gz / .mhd)
#
# Plus one of the following labeling modes:
#   A) class + centers + radii
#      - class: 1 for positive, 0 for negative
#      - centers: python-literal list of (z, y, x) tuples in world coordinates (mm)
#      - radii: python-literal list of nodule radii in mm
#   B) mask_path
#      - mask_path: instance segmentation mask path (background=0, each nodule id>=1)
#
# Optional columns:
#   - source_site, split, scanner_vendor, notes, ... (all are preserved in preprocess metadata)
MANIFEST_PATH = None

# if True, image_path inside the manifest is interpreted relative to RESOURCES_PATH.
MANIFEST_IMAGE_PATH_RELATIVE_TO_RESOURCES = True

# -------- Auto-discovery for NIfTI (.nii.gz) preprocessing --------
# If MANIFEST_PATH is None and AUTO_DISCOVER_NIFTI_GZ is True, preprocess will scan
# RESOURCES_PATH recursively for *.nii.gz and build an internal manifest.
AUTO_DISCOVER_NIFTI_GZ = False
NIFTI_GLOB_PATTERN = '**/*.nii.gz'
# Default class used for auto-discovered scans (0 means negative).
AUTO_DISCOVER_DEFAULT_CLASS = 0

# Save preprocessed volumes as compressed npz (image key: 'image').
PREPROCESSED_SAVE_FORMAT = 'npz'  # one of: 'npy', 'npz'

# -------- Instance-mask based target extraction (image.nii.gz + mask.nii.gz) --------
# If a manifest row contains mask_path, preprocess can derive class/centers/radii from
# mask instances automatically (background=0, each nodule instance has a unique id >=1).
MASK_PATH_RELATIVE_TO_RESOURCES = True

# In auto-discovery mode, if this suffix is set, preprocess will try to map each image
# to its mask path by replacing '.nii.gz' with this suffix.
# Example: image '/a/b/case001.nii.gz' + suffix '_mask.nii.gz' -> '/a/b/case001_mask.nii.gz'
AUTO_DISCOVER_MASK_SUFFIX = None

# For negative scans with empty mask instances, optionally create one synthetic center
# (scan center) so negative patch extraction still produces training samples.
AUTO_ADD_NEGATIVE_CENTER_FROM_IMAGE = True

# -------- Auto-manifest generation from image/label directories --------
# Set these two paths to automatically build manifest CSVs for image/label pair datasets
# where image and mask filenames match (both .nii.gz).
AUTO_GENERATE_MANIFEST_FROM_DIRS = True
IMAGE_DIR = "/home/ai/nnDetection-main/nndet_data/Task010_lung/raw_splitted/imagesTr"
LABEL_DIR = "/home/ai/nnDetection-main/nndet_data/Task010_lung/raw_splitted/labelsTr"
MANIFEST_OUTPUT_DIR = None  

# K-fold split settings used by manifest generation.
KFOLD_SPLITS = 5
KFOLD_SEED = 2026

# -------- Training options --------
BATCH_SIZE = 2
NUM_WORKERS = 0
PIN_MEMORY = True
PERSISTENT_WORKERS = False

PADDING_FOR_LOCALIZATION = 10
BLOCK_SIZE = 128
COORDS_CUBE_SIZE = 32
TARGET_SHAPE = (COORDS_CUBE_SIZE, COORDS_CUBE_SIZE, COORDS_CUBE_SIZE, 3, 5)
COORDS_SHAPE = (3, COORDS_CUBE_SIZE, COORDS_CUBE_SIZE, COORDS_CUBE_SIZE)
ANCHOR_SIZES = [10, 30, 60]
VAL_PCT = 0.2
TOTAL_EPOCHS = 100
DEFAULT_LR = 0.01
