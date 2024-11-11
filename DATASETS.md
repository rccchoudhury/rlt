# Dataset Installation Instructions

We provide detailed instructions for download and setting up the Kinetics-400 and Something-Something V2 datasets.

## Kinetics-400

To download the Kinetics-400 dataset, you can visit the following page: https://github.com/cvdfoundation/kinetics-dataset. The same process applies to Kinetics-600 and 700, but we only evaluated on and provide checkpoints for Kinetics-400.

We restate the instructions below:

1. Clone the Kinetics-400 repository:
    ```bash
    git clone https://github.com/cvdfoundation/kinetics-dataset.git
    cd kinetics-dataset
    ```

2. Download the Kinetics-400 video tarballs: 
    ```bash
    bash ./k400_downloader.sh
    ```

3. Extract the Kinetics-400 video tarballs:
    ```bash
    bash ./k400_extractor.sh
    ```

4. Download the Kinetics-400 annotations from the links on the page. You need to format the annotations in the following format: 

    ```bash
    /path/to/video_1 label_1
    /path/to/video_2 label_2
    ```
We provide a script for this (`scripts/make_annot_file.py`) to construct the annotation file from the video folder structure, where each video is placed in a folder with the same name as the label.

Once complete, make a directory in the main folder called `data` and move the extracted data into it. the project directory structure should look like this: 

```bash
.
├── rlt
│   ├── src
│   ├── data
│   │   ├── kinetics400
│   │   │   ├── train_labels.txt
│   │   │   ├── val_labels.txt
│   └── ...
```
To check that all the data is loading properly, we advise you run the `benchmark_dataloader.py` script:

```bash
python scripts/benchmark_dataloader.py
```

If this can iterate through the whole dataset, everything is set up properly!

### Something-Something V2
1. Please download the dataset and annotations from [dataset provider](https://20bn.com/datasets/something-something).

2. Download the frame list from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/val.csv)).

3. Run `scripts/reformat.py` to create the metadata file.

4. How to clean labels (from Sihan) (TODO)

### Setup Datasets Path
For ease of use, we recommend setting the following environment variables to avoid having to specify the data path in the config file.

#### Kinetics400
``` 
export KINETICS_TRAIN_METADATA=/yourpath/kinetics400_train.txt
export KINETICS_VAL_METADATA=/yourpath/kinetics400_val.txt
```
#### SSV2
```
export SSV2_TRAIN_ANNO=/yourpath/train.csv
export SSV2_VAL_ANNO=/yourpath/train.csv
```