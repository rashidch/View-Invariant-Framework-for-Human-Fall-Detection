import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence
class SeqDataset(Dataset):
                
    def __init__(self, inputX,tagY):
        self.X = inputX
        self.Y = tagY

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'data':torch.Tensor(self.X[idx]),'label':(self.Y[idx])}

    def collate_fn_pack(samples):
        datas = [s['data'] for s in samples]
        tags = [s['label']  for s in samples]
        datas = pack_sequence(datas,enforce_sorted=False)
        #print('datas len pad', len(datas), len(tags))
        return {'data':datas,'label':tags}

    def collate_fn_pad(samples,batch_first=True):
        datas = [s['data'] for s in samples]
        lens =  [len(d) for d in datas]
        tags = [s['label']  for s in samples]
        lens = torch.Tensor(lens)
        datas = pad_sequence(datas,batch_first=batch_first)
        #print('datas len pad', len(datas), len(tags))
        return {'data':(datas,lens),'label':tags}

class SinglePoseDataset(Dataset):            
    
    def __init__(self,dataset,labels):
        self.X = dataset
        self.Y = labels
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {'data':torch.Tensor(self.X[idx]),'label':(self.Y[idx])}
    