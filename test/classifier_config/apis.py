
def get_classifier_cfg(args):
    if args.classmodel.lower() == 'dnntiny':
        from test.classifier_config.act_dnntiny_cfg import cfg3
        return cfg3
    if args.classmodel.lower() == 'dnntiny3d':
        from test.classifier_config.act_dnntiny3D_cfg import cfg3
        return cfg3
    if args.classmodel.lower() == 'net':
        from test.classifier_config.net2d3d_cfg import cfg2
        return cfg2
    elif args.classmodel.lower() =='lstm2d':
        from test.classifier_config.act_lstmfc_cfg_2d import cfg3
        return cfg3
    elif args.classmodel.lower() =='lstm3d':
        from test.classifier_config.act_lstmfc_cfg_3d import cfg3
        return cfg3
    elif args.classmodel.lower() =='fallnet':
        from test.classifier_config.act_aelstm_cfg import cfg3 
        return cfg3
    raise NotImplementedError

    '''
    if args.classmodel == 'DNN_Single':
        from test.classifier_config.DNN_Single_cfg import cfg9 
        return cfg9
    if args.classmodel == 'DNN_':
        from test.classifier_config.DNN_cfg import cfg9 
        return cfg9
    if args.classmodel == 'DNN':
        from test.classifier_config.DNN2_cfg import cfg9 
        return cfg9
    '''
    

