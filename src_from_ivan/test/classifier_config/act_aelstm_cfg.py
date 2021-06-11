from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'FallNet'
cfg3.CHEKPT = 'checkpoints_fallmodel/aelstm/with_sigmoid/cpr15_Fall1-Seq_30/classSig_epoch_499.pth'
cfg3.tagI2W = ["Fall","Stand"]
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}