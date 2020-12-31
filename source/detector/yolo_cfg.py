from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'source/detector/yolo/cfg/yolov3-spp.cfg'
cfg.WEIGHTS = 'source/detector/yolo/data/yolov3-spp.weights'
cfg.INP_DIM =  320
cfg.NMS_THRES =  0.30
cfg.CONFIDENCE = 0.6
cfg.NUM_CLASSES = 80
