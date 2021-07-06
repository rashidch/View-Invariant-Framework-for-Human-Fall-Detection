from easydict import EasyDict as edict

#dnntiny cfg
cfg1 = edict()
cfg1.MODEL = 'dnntiny'
cfg1.CHEKPT = 'checkpoints/dnn_Jun_19_16/fold_4_epoch_0_loss_0.006082.pth'
cfg1.tagI2W = ["Fall","Stand"] # 2
cfg1.tagW2I = {w:i for i,w in enumerate(cfg1.tagI2W)}

#dnnet cfg
cfg2 = edict()
cfg2.MODEL = 'dnnnet'
cfg2.CHEKPT = 'checkpoints/alph_dnnnet_Jun_30_22/epoch_729_loss_0.041174.pth'
cfg2.tagI2W = ["Fall","Stand"] # 2
cfg2.tagW2I = {w:i for i,w in enumerate(cfg2.tagI2W)}

#2d3ddnnet cfg
cfg3 = edict()
cfg3.MODEL = 'net'
cfg3.CHEKPT = 'checkpoints/dnn2d3d_Jun_18/epoch_1919_loss_3.681156.pth'
cfg3.tagI2W = ["Fall","Stand"] # 2
cfg3.tagW2I = {w:i for i,w in enumerate(cfg3.tagI2W)}

#aelstm cfg
cfg4 = edict()
cfg4.MODEL = 'FallNet'
cfg4.CHEKPT = 'checkpoints_fallmodel/aelstm/with_sigmoid/cpr15_Fall1-Seq_30/classSig_epoch_499.pth'
cfg4.tagI2W = ["Fall","Stand"]
cfg4.tagW2I = {w:i for i,w in enumerate(cfg4.tagI2W)}

#lstmfc cfg
cfg5 = edict()
cfg5.MODEL = 'FallModel'
cfg5.CHEKPT = 'checkpoints/lstm2d_Jun_8/epoch_133_loss_0.024662.pth'
cfg5.tagI2W = ["Fall","Stand"]
cfg5.tagW2I = {w:i for i,w in enumerate(cfg5.tagI2W)}