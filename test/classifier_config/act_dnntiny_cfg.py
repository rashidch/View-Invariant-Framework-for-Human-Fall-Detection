from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'dnntiny'
cfg3.CHEKPT = 'checkpoints/dnn_Jun_19_16/fold_4_epoch_0_loss_0.006082.pth'
cfg3.tagI2W = ["Fall","Stand"] # 2
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}
