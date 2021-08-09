from fallModels.models import dnntiny,dnnnet,Net, FallModel, FallNet, GenNet
from fallModels.stmodel import DSTANet

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
    elif(name=='dstanet'):
        '''
        config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],
              [128, 256, 64, 2], [256, 256, 64, 1],
              [256, 256, 64, 1], [256, 256, 64, 1],]
        
        config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],]
        '''
        config = [[64, 64, 16, 1], [64, 64, 16, 1]]
        return DSTANet(num_class=2, num_point=11, num_frame=n_frames,config=config)

