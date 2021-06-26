from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'lstm3d'
cfg3.CHEKPT = 'checkpoints/lstm3d_Jun_13/epoch_135_loss_0.001917.pth'
cfg3.tagI2W = ["Fall","Stand"]
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}