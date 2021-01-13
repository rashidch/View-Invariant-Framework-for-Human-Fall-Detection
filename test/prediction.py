import numpy as np
import torch
from easydict import EasyDict as edict
from fallModels.F import normalize_min_
from fallModels.models import get_model
from test.classifier_config.apis import get_classifier_cfg
from test.detection_loader import DetectionLoader
"""----- Load modules from source for structring code ----"""

from source.alphapose.utils.transforms import get_func_heatmap_to_coord
from source.alphapose.utils.pPose_nms import pose_nms
from source.alphapose.utils.transforms import flip, flip_heatmap
from source.alphapose.models import builder
from source.detector.apis import get_detector

class classifier():
    def __init__(self, opt):

        self.opt   = opt
        self.cfg   = get_classifier_cfg(self.opt)
        self.model = None
        self.holder = edict()

    def load_model(self):
        self.model = get_model(self.cfg.MODEL,self.cfg.tagI2W)
        ckpt = torch.load(self.cfg.CHEKPT, map_location=self.opt.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if len(self.opt.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model
                , device_ids=self.opt.gpus).to(self.opt.device)
        else:
            self.model.to(self.opt.device)
        self.model.eval()
    
    def predict_action(self, keypoints):
        points = keypoints.numpy()
        points = normalize_min_(points)
        points = points.reshape(1,170)
        actres = self.model.exe(points,self.opt.device,self.holder)
        return actres

    def drawTagToImg(self,img, prediction):
        tag = self.cfg.tagI2W[prediction]
        img = cv2.putText(img, tag, (800,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        return img

class PoseEstimation():
       
    def __init__(self, args, cfg):
        
        self.args = args
        self.cfg = cfg

        # Load pose model
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        print(f'Loading pose model from {args.checkpoint}...')
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        self.pose_model.to(args.device)
        self.pose_model.eval()
        # Human detection model for cropping the person
        self.det_loader = DetectionLoader(get_detector(self.args), self.cfg, self.args)

    def process(self, im_name, image):
        try: 
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = self.det_loader.process(im_name, image).read()
                if orig_img is None:
                    raise Exception("no image is given")

                if boxes is None or boxes.nelement() == 0:
                        self.writer.save(None, None, None, None, None, orig_img, im_name)
                        pose = self.writer.start()

                # pose Estimation
                inps = inps.to(self.args.device) 
                if self.args.flip:
                    inps = torch.cat((inps, flip(inps)))
                hm = self.pose_model(inps)
                if self.args.flip:
                    hm_flip = flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                    hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
                hm = hm.cpu()


        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass
        except KeyboardInterrupt:
            print('===========================> Finish Model Running.')    

        
    

