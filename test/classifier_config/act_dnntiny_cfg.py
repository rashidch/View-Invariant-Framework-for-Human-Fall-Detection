from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'dnntiny'
cfg3.CHEKPT = 'checkpoints/dnntiny_May_13_10:19:46/epoch_139_loss_0.035889.pth'
cfg3.tagI2W = ["Fall","Stand"] # 2
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}
