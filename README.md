# PyTorch Implementation of Iterated Integrated Attributions

<p align="center">
  <img width="600" height="450" src="result_sample.png" alt="ViT" title="ViT">
</p>

## Introduction

This is the official PyTorch implementation of the Iterated Integrated Attributions (IIA) method.

We introduce a novel method that enables visualization of predictions made by vision models, as well as visualization of
explanations for a specific class.
In this method, we present the concept of iterated integrated attributions.

## Producing IIA Classification Saliency Maps

Images should be stored in the `data\ILSVRC2012_img_val` directory.
The information on the images being evaluated and their target class can be found in the `data\pics.txt` file, with each
line formatted as follows: `<file_name> target_class_number` (e.g. `ILSVRC2012_val_00002214.JPEG 153`).

To generate saliency maps using our method on CNN, run the following command:

```
python cnn_saliency_map_generator.py
```

And to produce maps for ViT, run:

```
python vit_saliency_map_generator.py
```

By default, saliency maps for CNNs are generated using the Resnet101 network on the last layer. You can change the
selected network and layer by modifying the `model_name` and `FEATURE_LAYER_NUMBER` variables in
the `cnn_saliency_map_generator.py` class. For ViT, the default is ViT-Base, which can also be configured using
the `model_name` variable in the `vit_saliency_map_generator.py` class.

The generated saliency maps will be stored in the `qualitive_results` directory.

### ViT models weight files:

-
ViT-B [Link to download](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)
-
ViT-S [Link to download](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth)

## Reproducing Segmentation Results

### Download the segmentaion datasets:

- Download
  imagenet_dataset [Link to download dataset](http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat)
- Download the COCO_Val2017 [Link to download dataset](https://cocodataset.org/#download)
- Download Pascal_val_2012 [Link to download dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

To run the image segmentation, download the VOC and COCO datasets to `data/VOC` and `data/COCO` respectively.
For Imagenet segmentation, use the dataset provided in the
link ([Link to download dataset](http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat)).
To run segmentation using ViT, configure the `chosen_dataset` variable in the `seg_vit_datasets` class (the default is
COCO) and run the following command:

```
python seg_vit_datasets.py
```

To run segmentation using CNN, specify the desired dataset `{dataset}` (which can be `imagenet`, `coco`, or `voc`) in
the following command and run it:

```
python segmentation_cnn_{dataset}.py
```

## Credits

For comparison, we used the following implementations of code from git repositories:

- https://github.com/jacobgil/pytorch-grad-cam
- https://github.com/pytorch/captum
- https://github.com/PAIR-code/saliency
