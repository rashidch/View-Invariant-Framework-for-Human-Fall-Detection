from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'FallModel'
cfg3.CHEKPT = 'checkpoints/lstm_Jun_8/epoch_134_loss_0.013666.pth'
cfg3.tagI2W = ["Fall","Stand"]
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}