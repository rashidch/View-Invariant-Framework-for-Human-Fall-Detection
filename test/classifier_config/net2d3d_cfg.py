from easydict import EasyDict as edict

cfg2 = edict()
cfg2.MODEL = 'net'
cfg2.CHEKPT = 'checkpoints/dnn2d3d_Jun_12_cam7/epoch_429_loss_3.245900.pth'
cfg2.tagI2W = ["Fall","Stand"] # 2
cfg2.tagW2I = {w:i for i,w in enumerate(cfg2.tagI2W)}
