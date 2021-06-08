import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def get_model(name,tagI2W, n_frames=5, pose2d_size=34, pose3d=51):
    if(name =='net'):
        #print("loaded Net model")
        return Net(pose2d_size,pose3d, len(tagI2W))
    elif(name == 'dnntiny'):
        print("loaded dnntiny model")
        return dnntiny(input_dim=pose2d_size*n_frames, class_num=len(tagI2W))
    elif(name == 'FallModel'):
        print("loaded FallModel model")
        return FallModel(input_dim=pose2d_size, class_num=len(tagI2W))
    elif(name == 'FallNet'):
        print("loaded FallNet model")
        return FallNet(input_dim=pose2d_size, class_num=len(tagI2W))
    elif(name == 'DNN'):
        print("loaded DNN model")
        return DNN(input_dim=pose2d_size*n_frames, class_num=len(tagI2W))
#dnn block
class block(nn.Module):
    def __init__(self,input_size):
        super(block,self).__init__()
        self.hidden = 1024
        self.fc1 = nn.Linear(input_size, self.hidden)
        self.bn1 = nn.BatchNorm1d(self.hidden)
        self.fc2 = nn.Linear(self.hidden,self.hidden)
        self.bn2 = nn.BatchNorm1d(self.hidden)
        
    def forward(self, _input):
        _fc1 = self.fc1(_input)
        _bn1 = self.bn1(_fc1)
        _fc1 = F.dropout(F.relu(_bn1),p=0.5) 
        _bn2 = self.bn2(self.fc2(_fc1))
        _fc2 = F.dropout(F.relu(_bn2), p=0.5)
        return _fc2
        
class dnn(nn.Module):
    def __init__(self, input_size):
        super(dnn,self).__init__()
        self.hidden = 1024
        self.block1 = block(input_size)
        self.block2 = block(self.hidden)
        
    def forward(self, _input):
        x_1 = self.block1(_input)
        x_2 = self.block2(x_1)
        x_3 = x_1 + x_2         
        return x_3

class Net(nn.Module):
    def __init__(self,input2d,input3d,class_num):
        super(Net,self).__init__()
        self.hidden = 1024
        self.dnn2d = dnn(input2d)
        self.dnn3d = dnn(input3d)
        self.fccom = torch.nn.Linear( self.hidden,self.hidden)
        self.bncom = torch.nn.BatchNorm1d( self.hidden)
        self.fc_cls = torch.nn.Linear( self.hidden, class_num)
        self.class_num = class_num
    
    def forward(self, input1, input2):
        out1 = self.dnn2d(input1)
        out2 = self.dnn3d(input2)
        out3 = out1 + out2
        _bncom = self.bncom(self.fccom(out3))
        _fccom = F.dropout(F.relu(_bncom), p=0.5)
        _fc_cls = self.fc_cls(_fccom)
        output = F.softmax(_fc_cls, dim=1)
        return out1, out2, _fc_cls, output
    
    def exe(self,input1_,input2_,device,holder):
        input1_ = torch.Tensor(input1_).to(device)
        input2_ = torch.Tensor(input2_).to(device)
        return self.__call__(input1_,input2_) # data,datalen


class dnntiny(torch.nn.Module):
    
    def __init__(self, input_dim, class_num):
        
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,120)
        #print('fc1',self.fc1)
        self.bn1 = torch.nn.BatchNorm1d(120)
        self.fc2 = torch.nn.Linear(120, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fc3 = torch.nn.Linear(64,32)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.fc4 = torch.nn.Linear(32, class_num)
        self.class_num = class_num
      
    def forward(self, _input):
        _fc1 = F.relu(self.fc1(_input))
        _bn1 = self.bn1(_fc1)
        _fc2 = F.relu(self.fc2(_bn1))
        _bn2 = self.bn2(_fc2)
        _fc3 = F.relu(self.fc3(_bn2))
        _bn3 = self.bn3(_fc3)
        _fc4 = self.fc4(_bn3)
        output = F.softmax(_fc4, dim=1)
        
        return _fc4, output

    def exe(self,input_,device,holder):
        input_ = torch.Tensor(input_).to(device)
        return self.__call__(input_)

#LSTM Model
class FallModel(torch.nn.Module):
    
    def __init__(self, input_dim, class_num):
        super(FallModel, self).__init__()
        self.input = input_dim
        self.num_layers = 3
        self.hidden_state = 34
        self.lstm = nn.LSTM(self.input, self.hidden_state, num_layers=self.num_layers, dropout=0.5 , batch_first=True)
        self.linear = nn.Sequential(nn.Linear(self.hidden_state, class_num), nn.ELU())
        self.class_num = class_num
    
    def forward(self, _input):
        
        feature ,_ = self.lstm(_input)
        raw_preds = self.linear(feature[:,-1,:])
        output_probs = F.softmax(raw_preds, dim=1)
        return raw_preds, output_probs

    def exe(self,input_,device,holder):
        input_ = torch.Tensor(input_).to(device)
        return self.__call__(input_)


class FallNet(nn.Module):
    def __init__(self, input_dim=24, class_num=2):
        super(FallNet, self).__init__()
        self.input = input_dim
        self.num_layers = 3
        self.hidden_state = 24
        self.lstm = nn.LSTM(self.input, self.hidden_state, self.num_layers, batch_first=True,dropout=0.5)
        self.linear = nn.Sequential(nn.Linear(self.hidden_state, 2),nn.ELU())
        self.class_num = class_num

    def forward(self, inputs):
        features,_ = self.lstm(inputs)
        raw_preds = self.linear(features[:,-1,:])
        output_probs = torch.sigmoid(raw_preds)
        return raw_preds, output_probs
    
    def exe(self,input_,device,holder):
        input_ = torch.Tensor(input_).to(device)
        return self.__call__(input_)


class GenNet(nn.Module):
    def __init__(self, Num):
        super(GenNet, self).__init__()

        self.num_layers = 3
        self.hidden_state = Num
        self.Enlstm = nn.LSTM(24, self.hidden_state, 2, batch_first=True, dropout=0.5)
        self.Delstm = nn.LSTM(self.hidden_state, 24, 2, batch_first=True, dropout=0.5)

    def forward(self, inputs):
        encoder, _ = self.Enlstm(inputs)
        decoder, _ = self.Delstm(encoder)
        return decoder
