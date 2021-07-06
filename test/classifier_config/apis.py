
def get_classifier_cfg(args):
    if args.classmodel.lower() == 'dnntiny':
        from test.classifier_config.act_dnntiny_cfg import cfg3
        return cfg3
    elif args.classmodel.lower() == 'dnnnet':
        from test.classifier_config.act_dnnnet_cfg import cfg2
        return cfg2
    elif args.classmodel.lower() == 'net':
        from test.classifier_config.net2d3d_cfg import cfg2
        return cfg2
    elif args.classmodel.lower() =='fallmodel':
        from test.classifier_config.act_lstmfc_cfg import cfg3 
        return cfg3
    elif args.classmodel.lower() =='fallnet':
        from test.classifier_config.act_aelstm_cfg import cfg3 
        return cfg3
    else:
        print('Model not implemented')
        raise NotImplementedError


def getFallModelcfg(args):
    if args.classmodel.lower() == 'dnntiny':
        from test.classifier_config.fallModelcfg import cfg1
        return cfg1
    elif args.classmodel.lower() == 'dnnnet':
        from test.classifier_config.fallModelcfg import cfg2
        return cfg2
    elif args.classmodel.lower() == 'net':
        from test.classifier_config.fallModelcfg import cfg3
        return cfg3
    elif args.classmodel.lower() =='fallmodel':
        from test.classifier_config.fallModelcfg import cfg5 
        return cfg5
    elif args.classmodel.lower() =='fallnet':
        from test.classifier_config.fallModelcfg import cfg5 
        return cfg5
    else:
        print('Model not implemented')
        raise NotImplementedError


