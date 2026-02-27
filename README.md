# LUNA16 object detection
Developing a well-documented repository for the Lung Nodule Detection task on the Luna16 dataset. This work is inspired by the ideas of the first-placed team at [DSB2017](https://www.kaggle.com/c/data-science-bowl-2017), "[grt123](https://github.com/lfz/DSB2017)".
<hr>
Lung cancer is the most common cancer in the world. More people die as a result of lung cancer each year than from breast, colorectal, and prostate cancer combined.
Lung nodule detection is an important process in detecting lung cancer. Lots of work has been done in providing a robust model, however, there isn't an exact solution by now. 

The [Data Science Bowl](https://datasciencebowl.com) (or DSB in short) is the world's premier data science for social good competition, created in 2014 and presented by Booz Allen Hamilton and Kaggle. The Data Science Bowl brings together data scientists, technologists, and domain experts across industries to take on the world's challenges with data and technology.
In DSB2017, the competition was held to find lung nodules. Team grt123 came up with [the best results](https://www.kaggle.com/c/data-science-bowl-2017/leaderboard) by implementing a 3d UNet based YOLO. At this repository, I'm going to attack the problem inspired by their approach and provide a better result in some cases. Furthermore, I'm going to make it well documented to enrich the literature.

Making their implementation working on every GPU configurations (even no GPU!), changing the data pre-processing, and augmentation from a big monotonic code into two stages are my first goals.
Mainly I have gone through [their paper](https://arxiv.org/abs/1711.08324).
I hope it helps researchers. If you have any questions on the code, please send an [email to me](mailto:s.mostafa.a96@gmail.com?subject=[GitHub]%20LUNA16%20grt123).

# Code description
## Prepare
The preparation code is implemented in the `prepare` package, including pre-processing and augmentation.

### Tutorials:
There is a [jupyter notebook as a tutorial](./notebooks/Preprocessor.ipynb), covering the pre-processing steps. 
Take a look at the `prepare._classes.CTScan.preprocess` method. 
Also, there is another [jupyter notebook tutorial](./notebooks/Augmentor.ipynb) explaining the augmentation techniques of the method `prepare._classes.PatchMaker._get_augmented_patch`, 
which is strongly recommended to review.

## Model
In order to have a good image of the "Nodule Net", you could study [the paper]!
The below image shows the network structure. 
Its code is in model package mostly the same as the original version of the code.
Also, loss computation at `model/loss.py` is an IOU approach, to know the details you can read their paper.
![Net](./notebooks/figs/net.png)

## Main
The `LunaDataSet` class in `main/dataset.py`, loads the saved augmented data to a torch `Dataset` and uses it to form a `DataLoader` and then feed the model as well as computing the loss.

# How to use
1. Download the Luna16 dataset from [here](http://academictorrents.com/collection/luna-lung-nodule-analysis-16---isbi-2016-challenge).
**There is also a small version of the dataset just for testing which is available in my google drive [here](https://drive.google.com/file/d/1QOSRnUiwp08AFYOFgrCWJrEEEckZG1_0/view?usp=sharing), it is because the size of the original dataset is too large to download.**
Also, for more information, the dataset description is available [here](https://luna16.grand-challenge.org/data/).
2. Change the first 2 variables in `configs.py` file

3. Run `python -m prepare.run_preprocess`

4. Run `python -m prepare.run_augmentation`

5. Run `python -m main.train`

> Tip: Prefer module-style commands (e.g., `python -m prepare.run_preprocess`) to avoid `ModuleNotFoundError: configs` caused by script-path imports.

> ⚠️ If you ever see `ModuleNotFoundError: configs`, do **not** install `configs` from PyPI.
> This project expects the local `configs.py` file in the repo root; just run commands from the repo and prefer `python -m ...`.


### Using google colab
The model has been trained in some small epochs by a [small sample](https://drive.google.com/file/d/1QOSRnUiwp08AFYOFgrCWJrEEEckZG1_0/view?usp=sharing) on google colab infrastructure.
You could simply copy the data to your own Google Drive account and run [this notebook](./notebooks/Sample_of_training_process_with_google_colab.ipynb) to learn the procedure of training the model using google colab!

## Training with custom `.nii.gz` datasets (industrial workflow)
The preprocessing script now supports two modes:

1. **Legacy LUNA16 mode** (default):
   - Uses `RESOURCES_PATH/annotations.csv`, `RESOURCES_PATH/candidates.csv`, and `*.mhd` files.
2. **Manifest mode** (recommended for production):
   - Set `MANIFEST_PATH` in `configs.py` and provide a CSV with at least:
     - `case_id`: unique case id
     - `image_path`: path to image (`.nii.gz` / `.mhd`)
     - `class`: `1` for positive, `0` for negative
     - `centers`: Python-literal list of world-coordinate tuples `(z, y, x)`
     - `radii`: Python-literal list of radii (mm)

Example manifest row:

```csv
case_id,image_path,class,centers,radii,split,source_site
case_0001,images/case_0001.nii.gz,1,"[(120.4, 256.0, 311.7)]","[4.5]",train,hospital_a
```

Then run exactly the same pipeline:

```bash
python -m prepare.run_preprocess
python -m prepare.run_augmentation
python -m main.train
```

### Recommended config updates for large-scale training
- Set `BATCH_SIZE`, `NUM_WORKERS`, `PIN_MEMORY`, `PERSISTENT_WORKERS` in `configs.py`.
- **Important**: training loop now iterates through all batches in an epoch (the previous single-batch debug break has been removed).
- Keep split by case/series (already done in `main/train.py`) to avoid train/val leakage.


### Auto-processing `.nii.gz` to `.npz`

If you do not want to provide a manifest, you can enable automatic NIfTI discovery in `configs.py`:

- `AUTO_DISCOVER_NIFTI_GZ = True`
- `NIFTI_GLOB_PATTERN = "**/*.nii.gz"`
- `PREPROCESSED_SAVE_FORMAT = "npz"`

Then run:

```bash
python -m prepare.run_preprocess
```

The preprocessed scans will be saved under:
- `OUTPUT_PATH/preprocessed/positives/*.npz`
- `OUTPUT_PATH/preprocessed/negatives/*.npz`

Each `.npz` stores the preprocessed 3D array under key `image`.


### Using image + instance-mask (`.nii.gz` + `.nii.gz`)

If your labels are instance masks (`0=background`, `1..N=each nodule instance`), you can let preprocess derive
`class/centers/radii` automatically from `mask_path`.

Manifest example:

```csv
case_id,image_path,mask_path,split
case_0001,images/case_0001.nii.gz,masks/case_0001_mask.nii.gz,train
case_0002,images/case_0002.nii.gz,masks/case_0002_mask.nii.gz,val
```

How it works:
- For each instance id in `mask_path`, preprocess computes:
  - center: centroid converted to world coordinate `(z,y,x)`
  - radius: equivalent-sphere radius from instance voxel volume
- If no instance exists, class is treated as negative (`class=0`).
- For negative masks, optional `AUTO_ADD_NEGATIVE_CENTER_FROM_IMAGE=True` adds one synthetic center at image center so
  negative patches can still be generated in augmentation.

Related config fields:
- `MASK_PATH_RELATIVE_TO_RESOURCES`
- `AUTO_ADD_NEGATIVE_CENTER_FROM_IMAGE`
- optional auto pairing in discovery mode: `AUTO_DISCOVER_MASK_SUFFIX`


### Auto-generate manifest + 5-fold split from `image/` and `labels/`

If your data is organized as:
- `image/*.nii.gz`
- `labels/*.nii.gz`

and filenames match one-to-one, you can auto-generate manifests without manually writing CSV files.

Set in `configs.py`:
- `AUTO_GENERATE_MANIFEST_FROM_DIRS = True`
- `IMAGE_DIR = "/path/to/image"`
- `LABEL_DIR = "/path/to/labels"`
- `KFOLD_SPLITS = 5`

Then run:

```bash
python -m prepare.run_preprocess
```

Generated files (default: `OUTPUT_PATH/manifests`):
- `manifest_all.csv`
- `manifest_fold0.csv` ... `manifest_fold4.csv`
- `manifest_fold0_train.csv` / `manifest_fold0_val.csv` ...
- `manifest_summary.csv`

Each generated row contains `case_id,image_path,mask_path` (and split columns for fold files).
Preprocess will also continue and convert scans to preprocessed files (`.npz`/`.npy` per config).
