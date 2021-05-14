import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def get_model(name,tagI2W, n_frames=5, POSE_JOINT_SIZE=24):
    if(name == 'dnnSingle'):
        return dnnSingle(POSE_JOINT_SIZE,len(tagI2W))
    elif(name == 'dnntiny'):
        print("loaded dnntiny model")
        return dnntiny(input_dim=POSE_JOINT_SIZE*n_frames, class_num=len(tagI2W))
    elif(name == 'FallModel'):
        print("loaded FallModel model")
        return FallModel(input_dim=POSE_JOINT_SIZE, class_num=len(tagI2W))
    elif(name == 'FallNet'):
        print("loaded FallNet model")
        return FallNet(input_dim=POSE_JOINT_SIZE, class_num=len(tagI2W))
    elif(name == 'DNN'):
        print("loaded DNN model")
        return DNN(input_dim=POSE_JOINT_SIZE*n_frames, class_num=len(tagI2W))

class dnnSingle(nn.Module):
    def __init__(self,input_dim,class_num,batch_first=True,initrange=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, class_num)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(16)
        self.class_num = class_num
        
        self.init_weights(initrange)
        self.batch_first = batch_first

    def init_weights(self,initrange):
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        self.fc4.weight.data.uniform_(-initrange, initrange)
        self.fc4.bias.data.zero_()

    def forward(self, _input):
        _fc1   = F.relu(self.fc1(_input))
        _bn1 = self.bn1(_fc1)
        _fc2   = F.relu(self.fc2(_bn1))
        _bn2 = self.bn2(_fc2)
        _fc3   = F.relu(self.fc3(_bn2))
        _bn3 = self.bn3(_fc3)
        _fc4   = F.relu(self.fc4(_bn3))

        _out = F.softmax(_fc4,dim=1) #single
        
        return _out.view(-1,_out.shape[-1])

    def exe_pre(self,device,holder):
        pass

    def exe(self,input_,device,holder):
        input_ = torch.Tensor(input_).to(device)
        return self.__call__(input_) # data,datalen


class DNN_Single(torch.nn.Module):
    
    def __init__(self, input_dim, class_num, initrange=0.5):
        
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fc3 = torch.nn.Linear(64,16)
        self.bn3 = torch.nn.BatchNorm1d(16)
        self.fc4 = torch.nn.Linear(16, class_num)
        self.class_num = class_num
        self.init_weights(initrange)
    
    def init_weights(self, initrange):
        
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        self.fc4.weight.data.uniform_(-initrange, initrange)
        self.fc4.bias.data.zero_()
    
    def forward(self, _input):
        
        _fc1 = F.relu(self.fc1(_input))
        
        _bn1 = self.bn1(_fc1)
        
        _fc2 = F.relu(self.fc2(_bn1))
        _bn2 = self.bn2(_fc2)
        
        _fc3 = F.relu(self.fc3(_bn2))
        _bn3 = self.bn3(_fc3)
        
        _fc4 = self.fc4(_bn3)
        
        output = F.softmax(_fc4, dim=1)
        
        return output

    def exe(self,input_,device,holder):
        input_ = torch.Tensor(input_).to(device)
        return self.__call__(input_) # data,datalen


class DNN(torch.nn.Module):
    
    def __init__(self, input_dim, class_num, initrange=0.5, drop_p=0.8):
        
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,1024)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fc3 = torch.nn.Linear(512,256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fc4 = torch.nn.Linear(256,128)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.fc5 = torch.nn.Linear(128, class_num)
        self.class_num = class_num
        self.init_weights(initrange)
        self.p = drop_p
    
    def init_weights(self, initrange):
        
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        self.fc4.weight.data.uniform_(-initrange, initrange)
        self.fc4.bias.data.zero_()
        self.fc5.weight.data.uniform_(-initrange, initrange)
        self.fc5.bias.data.zero_()
    
    def forward(self, _input):
        
        _bn1 = self.bn1(self.fc1(_input))
        _fc1 = F.dropout(F.relu(_bn1),p=self.p) 
        
        _bn2 = self.bn2(self.fc2(_bn1))
        _fc2 = F.dropout(F.relu(_bn2), p=self.p)
        
        _bn3 = self.bn3(self.fc3(_fc2))
        _fc3 = F.dropout(F.relu(_bn3), p=0.5)
        
        _bn4 = self.bn4(self.fc4(_fc3))
        _fc4 = F.dropout(F.relu(_bn4), p=0.5)
        
        _fc5 = self.fc5(_fc4)
        
        output = F.softmax(_fc5, dim=1)
        
        return output
    
    def exe(self,input_,device,holder):
        input_ = torch.Tensor(input_).to(device)
        return self.__call__(input_) # data,datalen


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
        #print('input', _input.shape)
        _fc1 = F.relu(self.fc1(_input))
        #print('fc1:res',_fc1.shape)
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
        self.hidden_state = 24
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
