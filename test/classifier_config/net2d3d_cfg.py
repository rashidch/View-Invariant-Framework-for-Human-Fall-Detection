from easydict import EasyDict as edict

cfg2 = edict()
cfg2.MODEL = 'net'
cfg2.CHEKPT = 'checkpoints/dnn2d3d_Jun_7_Yungtay_NotAugmented/epoch_486_loss_3.410847.pth'
cfg2.tagI2W = ["Fall","Stand"] # 2
cfg2.tagW2I = {w:i for i,w in enumerate(cfg2.tagI2W)}
