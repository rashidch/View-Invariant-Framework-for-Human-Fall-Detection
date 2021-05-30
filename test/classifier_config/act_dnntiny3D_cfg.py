from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'dnntiny3d'
cfg3.CHEKPT = 'checkpoints/dnntiny_3D_May_30/epoch_78_loss_0.044305.pth'
cfg3.tagI2W = ["Fall","Stand"] # 2
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}
