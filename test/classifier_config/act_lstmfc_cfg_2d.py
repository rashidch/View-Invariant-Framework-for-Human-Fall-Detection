from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'lstm2d'
cfg3.CHEKPT = 'checkpoints/lstm2_Jun_16_Yungtay_NotAugmented/epoch_146_loss_0.016376.pth'
cfg3.tagI2W = ["Fall","Stand"]
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}