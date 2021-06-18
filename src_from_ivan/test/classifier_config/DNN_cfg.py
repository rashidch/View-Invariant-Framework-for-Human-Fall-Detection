from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'DNN_'
cfg3.CHEKPT = 'checkpoints_fallmodel/act_dnnSingle_5/epoch_2999.pth'
cfg3.tagI2W = ["Fall","Stand", "Tie"] # 3
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}

# cfg.CHEKPT = 'model/act_dnnSingle_5/epoch_1000.pth'
# cfg.tagI2W = ["jump","run","sit","stand","walk"]