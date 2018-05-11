# openimage_dataset_utils
Contain utilities used to work with openimage dataset

## Prepare data
Download data from [here](https://storage.googleapis.com/openimages/web/download.html).
You need to download:
[train-annotations-bbox.csv](https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv) which contains bound boxes with image ids and labelname.
[train-images-boxable.csv](https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable.csv) which contains image id to download url map.
You have to download data to `data` folder.

## Download images
Use the following command to download images:
```python utils/download_selected_classes.py ```

## Convert to tfrecord format

## Run training

## Run inference
