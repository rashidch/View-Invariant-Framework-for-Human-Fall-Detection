# Falling-Person-Detection-based-On-AlphaPose
A PyTorch implementation for Falling person detection inside elevator based on AlphaPose Estimation (Human body keypoints detection)

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->
## Table of Contents:
* [Falling-Person-Detection-based-On-AlphaPose](#Falling-Person-Detection-based-On-AlphaPose)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
    * [Fall Detection Results](#Inference-results)
      * Results on Custom Elevator Fall Detection Datatset
      * Results on Benchmark Datasets 
        * Le2i
        * MultiCam
    * [Contribution](#contribution)
    * [TODOs](#todos)
    * [License](#license)
    * [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->


## Features
* Clear folder structure which is suitable for many deep learning projects.
* Config files to support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.

## Folder Structure
  ```
  Falling-Person-Detection-based-On-AlphaPose/
  │
  ├── train/ contains training scripts for dnn and RNNS fall classification models
  |    ├──  train_dnn.py - script to start dnn training
  |    ├──  train_ae.py - script to start auto encoder training
  |    ├──  train_aelstm.py - script to start auto encode plus lstm training
  |    ├──  train_lstm.py - script to start lstm training
  |    ├──  dataloader.py - skeleton dataloader
  |    ├──  plot_statics.py - plot training stats 
  |
  ├── test/ contains  scripts to inference trained models on videos and image data
  |    ├── main.py - main script to run inference on video data
  |
  ├── source/contains alphapose source code  
  |    ├── check alphapose source readme for getting started
  |    ├── check alaphapose docs folder for installation 
  │
  │
  ├── dataset/  - contains skeleton data extracted from alphapose estimator 
  │   └── DataCSV
  |    └── DataPrepare 
  |    └── SkeletonData
  │
  ├── input/ - default directory for storing image and video datasets
  |	└── multicam_dataset - these are used as input to pose estimator for extracting skeleton data
  |	└── Falling_Standing
  |	└── Falling_Standing_2
  │
  ├── examples - contains test video for inferece 
  |
  ├── fallmodels/ - fall classification models package
  │   ├── model.py
  │     
  │
  ├── checkpoints/ fall classification models checkpoints
  │   ├── dnntiny/ - trained models are saved here
  |          ├── epoch_210_loss_0.031925.pth
  │
  ├── plots/ - module for tensorboard visualization and logging
  │   
  │  
  └── tools/ - small utility functions
      ├── cut_frames.py
      └── video2images.py
  ```
## Requirements
* Python >= 3.5 (3.7 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* See alphapose [readme](https://github.com/rashidch/Falling-Person-Detection-based-On-AlphaPose/tree/main/source) 
	and [installation docs](https://github.com/rashidch/Falling-Person-Detection-based-On-AlphaPose/blob/main/source/docs/INSTALL.md) for complete requirements
* After complete installation including Alphapose cd to root directory (Falling-Person-Detection-based-On-AlphaPos) and follow commands in usage section to extract sekelton data, run train and inference on videos

<p align='center'>
   <img src="/outputs/19O2hUEge-6tZ6N9dUYSO6l3cROvlhs5A", width="360">  
 </p>
	
## Usage

* Extract skeleton data:
  ```
  python dataset/dataPrepare/get_keypoints.py --cfg source/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint source/pretrained_models/fast_res50_256x192.pth --indir input/Falling_Standing_2 --outdir frames --save_img --qsize 50
  ```
  <p align='center'>
   <img src="/outputs/1_PUhxNG8HKGvaquMtFjtuLpRfGJwb_mF", width="360">
   </p>
* Train fall classification models
  ```
  python train/train_dnn.py
  ```
* Run on trained fall models
  ```
  python test/main.py --cfg source/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint source/pretrained_models/fast_res50_256x192.pth --cam examples/demo/test/1.mp4 --vis_fast --save_out outputs/1.avi
  ```
* Check [alphapose docs](https://github.com/rashidch/Falling-Person-Detection-based-On-AlphaPose/blob/main/source/docs/run.md) for explanation of command line arguments 

### Config file format
* [Alphapose Config files ](https://github.com/rashidch/Falling-Person-Detection-based-On-AlphaPose/blob/main/source/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml)
* [Fall classfication models Config files ](https://github.com/rashidch/Falling-Person-Detection-based-On-AlphaPose/tree/main/test/classifier_config)
* Add addional configurations if you need.

## Inference results
   ### Falling and Standing Demo on Custom Dataset
   <p align='center'>
   <img src="/outputs/dnntiny/falling.gif", width="360">  
   <img src="/outputs/dnntiny/standing.gif", width="360">
   </p>
   
	
	

## Contribution
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs

- [ ] Training on Benchmark Datasets
- [ ] Testing on Benchmark Datasets
- [ ] Inference on Videos 


## License
This project is licensed under the MIT License. See  LICENSE for more details


## Acknowledgements
This project is developed based on Alphapose [codebase](https://github.com/MVIG-SJTU/AlphaPose).

