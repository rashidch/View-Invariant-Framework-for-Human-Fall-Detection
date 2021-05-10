# [Falling-Person-Detection-based-On-AlphaPose] (https://github.com/rashidch/Falling-Person-Detection-based-On-AlphaPose)
Falling person detection inside elevator based on AlphaPose Estimation (Human body keypoints detection)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

 An real-time Human Fall detection application for Elevator webcam. https://github.com/rashidch/Falling-Person-Detection-based-On-AlphaPose

## setup

### Requirements

- Linux
- Python 3.6.10+
- Cython
- Pytorch 1.1.0
- torchvision 0.3.0
- find (GNU findutils) 4.7.0
- viewnior 1.7
- AlphaPose

### Install Scripts

```
#install conda

#create env
conda create -n alphapose python=3.6 -y
conda activate alphapose

# install dependents
conda install pytorch==1.1.0 torchvision==0.3.0
conda install opencv
conda install jupyter notebook


pip install -r requirement.txt

#AlphaPose
#ToDo Alphapose installation details 


```

### Models

Download models from according to each `get_model.txt` in `\model` and make them look like:

```
.
├── act_dnnSingle_9
│   ├── epoch_1000.pth
│   └── epoch_500.pth
├── act_fcLstm_9
│   ├── epoch_1000.pth
│   └── epoch_500.pth
├── detector
│   ├── tracker
│   │   ├── jde.1088x608.uncertainty.pt
│   │   └── yolov3.cfg
│   └── yolo
│       ├── yolov3-spp.cfg
│       └── yolov3-spp.weights
├── pose_res152_duc
│   ├── 256x192_res152_lr1e-3_1x-duc.yaml
│   └── fast_421_res152_256x192.pth
├── pose_res50
│   ├── 256x192_res50_lr1e-3_1x.yaml
│   └── fast_res50_256x192.pth
└──  pose_res50_dcn
    ├── 256x192_res50_lr1e-3_2x-dcn.yaml
    └── fast_dcn_res50_256x192.pth
```


```

modify `templates/index.html` , add `/static`  to all resources link,like:

```html
origin:
<link rel=icon href=/favicon.ico>
modified:
<link rel=icon href=/static/favicon.ico>
```

## Run

For all arguments detail, see`config/default_args`.

For minimal scripts to run, see `local_demo.py`. And simply run by `python local_demo.py` in conda env.



## Train

models define in `actRec/models.py`

For train process and result,see [train/train-dnn.ipynb](train/train-dnn.ipynb) and [train/train-fclstm.ipynb](train/train-fclstm.ipynb).

Open and run them by `jupyter notebook` in conda env.

## License

```
Copyright 2019 github@livin2

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

[AlphaPose License]( https://github.com/MVIG-SJTU/AlphaPose/blob/master/LICENSE )
