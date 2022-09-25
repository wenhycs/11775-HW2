# CMU 11-775 Fall 2022 Homework 2

[PDF Handout](docs/handout.pdf)

In this homework we will perform a video classification task with visual features.

## Recommended Hardware

This code template is built based on [PyTorch](https://pytorch.org) and [Pyturbo](https://github.com/CMU-INF-DIVA/pyturbo) for Linux to fully utilize the computation of multiple CPU cores and GPUs.
SIFT feature, K-Means, and Bag-of-Words must run on CPUs, while CNN features and MLP classifiers can run on GPUs.
For AWS, a `g4dn.4xlarge` instance should be sufficient for the full pipeline.
During initial debugging, you are recommended to use a smaller instance to save money, e.g., `g4dn.xlarge` or a CPU-only equivalent for the SIFT part.
For more about AWS, see this [Doc](https://docs.google.com/document/d/1XkpGSzInT5TJz0hc0jUd7j5kGvuGO_wTOATW8pp4GCg/edit?usp=sharing) (Andrew ID required).

## Install Dependencies

A set of dependencies is listed in [environment.yml](environment.yml). You can use `conda` to create and activate the environment easily.

```bash
# Start from within this repo
conda env create -f environment.yml -p ./env
conda activate ./env
```

## Dataset

You will continue using the data from [Homework 1](https://github.com/KevinQian97/11755-ISR-HW1#data-and-labels) for this homework, which you should have downloaded.

If you don't have the data, download it from [AWS S3](https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data.zip) with the following commands:

```bash
# Start from within this repo
cd ./data
# Download and decompress data (no need if you still have it from HW1)
wget https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data.zip
unzip 11775_s22_data.zip
rm 11775_s22_data.zip
```

Eventually, the directory structure should look like this:

* this repo
  * code
  * data
    * videos (unzipped from 11775_s22_data.zip)
    * labels (unzipped from 11775_s22_data.zip)
  * env
  * ...

## Development and Debugging

Some functions in the pipeline are deliberately left blank for you to implement, where an `NotImplementedError` will be raised.
We recommend you generate a small file list (e.g. `debug.csv` with 20 lines) for fast debugging during initial development.
The `--debug` option in some scripts are also very helpful.
In addition, you can enable `pdb` debugger upon exception

```bash
# Instead of 
python xxx.py yyy zzz
# Run
ipython --pdb xxx.py -- yyy zzz
```

## SIFT Features

To extract SIFT features, use

```bash
python code/run_sift.py data/labels/xxx.csv
```

By default, features are stored under `data/sift`.

To train K-Means with SIFT feature for 128 clusters, use

```bash
python code/train_kmeans.py data/labels/xxx.csv data/sift 128 sift_128
```

By default, model weights are stored under `data/kmeans`.

To extract Bag-of-Words representation with the trained model, use

```bash
python code/run_bow.py data/labels/xxx.csv sift_128 data/sift
```

By default, features are stored under `data/bow_<model_name>` (e.g., `data/bow_sift_128`).

## CNN Features

To extract CNN features, use

```bash
python code/run_cnn.py data/labels/xxx.csv
```

By default, features are stored under `data/cnn`.

## 3D CNN Features

To extract 3D CNN features, use

```bash
python code/run_cnn3d.py data/labels/xxx.csv
```

By default, features are stored under `data/cnn3d`.

## MLP Classifier

The training script automatically and deterministically split the `train_val` data into training and validation, so you do not need to worry about it.

To train MLP with SIFT Bag-of-Words, run

```bash
python code/run_mlp.py sift --feature_dir data/bow_sift_128 --num_features 128
```

To train MLP with CNN features, run

```bash
python code/run_mlp.py cnn --feature_dir data/cnn --num_features <num_feat>
```

By default, training logs and predictions are stored under `data/mlp/model_name/version_xxx/`.
You can directly submit the CSV file to [Kaggle](https://www.kaggle.com/competitions/hw2-video-based-med/overview).
