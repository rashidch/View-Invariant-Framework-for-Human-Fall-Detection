from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'FallModel'
cfg3.CHEKPT = 'checkpoints/lstm2d_Jun_8/epoch_133_loss_0.024662.pth'
cfg3.tagI2W = ["Fall","Stand"]
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}