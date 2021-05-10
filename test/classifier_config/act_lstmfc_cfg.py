from easydict import EasyDict as edict

cfg3 = edict()
cfg3.MODEL = 'FallModel'
cfg3.CHEKPT = 'checkpoints_fallmodel/act_fclstm_5/epoch_237_loss_0.009316.pth'
cfg3.tagI2W = ["Fall","Stand", "Tie"]
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}