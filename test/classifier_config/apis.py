
def get_classifier_cfg(args):
    if args.classmodel.lower() == 'dnnsingle9':
        from test.classifier_config.act_dnnsingle_cfg import cfg9 
        return cfg9
    elif args.classmodel.lower() == 'fclstm9':
        from test.classifier_config.act_fclstm_cfg import cfg9 
        return cfg9
    if args.classmodel == 'DNN_Single':
        from test.classifier_config.DNN_Single_cfg import cfg9 
        return cfg9
    if args.classmodel == 'DNN_':
        from test.classifier_config.DNN_cfg import cfg9 
        return cfg9
    if args.classmodel == 'DNN':
        from test.classifier_config.DNN2_cfg import cfg9 
        return cfg9

    raise NotImplementedError

