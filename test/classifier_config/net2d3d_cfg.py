from easydict import EasyDict as edict

cfg2 = edict()
cfg2.MODEL = 'Net'
cfg2.CHEKPT = 'checkpoints/dnn2d3d_May_18_15:52:18/epoch_338_loss_3.464623.pth'
cfg2.tagI2W = ["Fall","Stand"] # 2
cfg2.tagW2I = {w:i for i,w in enumerate(cfg2.tagI2W)}
