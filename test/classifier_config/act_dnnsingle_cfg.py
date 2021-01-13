from easydict import EasyDict as edict

cfg9 = edict()
cfg9.MODEL = 'dnnSingle'
cfg9.CHEKPT = '/home/rasho/AlphaPose/model/act_dnnSingle3_2020-12-28_20-56/epoch_3000.pth'
cfg9.tagI2W = ["Fall","Stand", "Tie"] # 9
cfg9.tagW2I = {w:i for i,w in enumerate(cfg9.tagI2W)}

# cfg.CHEKPT = 'model/act_dnnSingle_5/epoch_1000.pth'
# cfg.tagI2W = ["jump","run","sit","stand","walk"]