# CurveReg

## Installation
1. Create a new Conda environment:

   ```bash
   conda create -n myenv python=3.10
   ```

2. Activate the environment:

   ```bash
   conda activate myenv
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```
## Dataset Structure
Your dataset should follow the following structure:
```text
├── Stain 1
│   └── Slides    # The slide images (e.g. .tiff, .svs)
│   └── Cutlines  # The binary mask images of cutlines for each slide, as specified in the paper
│   └── ROIs      # The binary mask images of the region-of-interest for each slide, to avoid processing unwanted areas (optional)
│   └── Masks     # The binary mask images of the landmarks for each slide
└── Stain 2
    └── ...
```
Note that *all files related to the same slide should have the same basename*, for example for a specific slide, the following naming convention is required:
- Slides/slide_1.tiff
- Cutlines/slide_1.png
- ROIs/slide_1.png
- Masks/slide_1.png

In additional to the dataset, each pair of stain to be matched requires an additional csv file, which contains additional information needed for each pair. 
See `train.ipynb` or `evaluation.ipynb` for more information, which shows what column is required for which part of the pipeline.

## Training
```
train.py [-h] -d1 DATASET_1 -d2 DATASET_2 -c CSV -o OUTPUT

options:
  -h, --help                               Show the help message and exit
  -d1 DATASET_1, --dataset-1 DATASET_1     Path to the reference/target dataset
  -d2 DATASET_2, --dataset-2 DATASET_2     Path to the dataset to be warped
  -c CSV, --csv CSV                        Path to the additional .csv file provided as part of the dataset
  -o OUTPUT, --output OUTPUT               Path to the output model
```
See `train.ipynb` for a broken down version of `train.py`.

## Training
```
evaluate.py [-h] -d1 DATASET_1 -d2 DATASET_2 -c CSV -m MODEL

options:
  -h, --help                               Show the help message and exit
  -d1 DATASET_1, --dataset-1 DATASET_1     Path to the reference/target dataset
  -d2 DATASET_2, --dataset-2 DATASET_2     Path to the dataset to be warped
  -c CSV, --csv CSV                        Path to the additional .csv file provided as part of the dataset
  -m MODEL, --model MODEL                  Path to the model
```
See `evaluation.ipynb` for a broken down version of `evaluation.py`. Note that the evaluation of TRE is based on the input pixel, not microns.
