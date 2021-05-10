from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'dnnSingle'
cfg3.CHEKPT = 'checkpoints_fallmodel/act_dnntiny_5/epoch_317_loss_0.012371.pth'
cfg3.tagI2W = ["Fall","Stand", "Tie"] # 9
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}
