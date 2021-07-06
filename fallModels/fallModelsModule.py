from fallModels.models import dnntiny,dnnnet,Net, FallModel, FallNet, GenNet


def getModel(name,tagI2W, n_frames=5, pose2d_size=34, pose3d=51):
    if(name =='net'):
        return Net(pose2d_size,pose3d, len(tagI2W))
    elif(name == 'dnntiny'):
        return dnntiny(input_dim=pose2d_size*n_frames, class_num=len(tagI2W))
    elif(name == 'FallModel'):
        return FallModel(input_dim=pose2d_size, class_num=len(tagI2W))
    elif(name == 'FallNet'):
        return FallNet(input_dim=pose2d_size, class_num=len(tagI2W))
    elif(name == 'dnnnet'):
        return dnnnet(input_dim=pose2d_size*n_frames, class_num=len(tagI2W))