from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'dnntiny'
cfg3.CHEKPT = 'checkpoints/dnntiny_May_14_14:34:53/epoch_210_loss_0.031925.pth'
cfg3.tagI2W = ["Fall","Stand"] # 2
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}
