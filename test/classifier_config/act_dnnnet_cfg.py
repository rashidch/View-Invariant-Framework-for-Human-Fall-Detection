from easydict import EasyDict as edict

cfg2 = edict()
cfg2.MODEL = 'dnnnet'
cfg2.CHEKPT = 'checkpoints/alph_dnnnet_Jun_30_22/epoch_729_loss_0.041174.pth'
cfg2.tagI2W = ["Fall","Stand"] # 2
cfg2.tagW2I = {w:i for i,w in enumerate(cfg2.tagI2W)}
