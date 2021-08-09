"""
Purposes of this code is to create pickle label for every video inside ../examples/demo/test/
"""

import glob
import pickle

path = 'C:\\Users\\rashi\\Documents\\FallDataset\\FallDataset_2\\le2i_annotated\\testData'
files = glob.glob(path+'/*/*png')

#print(files)
dictionary={}
tmp=None
for file in files:
    files_summary=file.split("\\")
    #video Name = framefoldername = labelfilename e.g. trainData.pickle
    dictfile_name=files_summary[-3]
    #Frame_Id e.g. 00000.png --> 00000
    frameId=files_summary[-1].split('.')[0]
    #Label folderName --> Fall or NotFall 
    label=files_summary[-2]
    #print(files_summary)
    print(dictfile_name,frameId,label)
    if(tmp==dictfile_name):
        dictionary[frameId]=label
    elif(tmp==None):
        tmp=dictfile_name
        dictionary[frameId]=label
    else:
        with open(path+'\\'+tmp+'.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        dictionary={}
        dictionary[frameId]=label
        print(tmp)
        tmp=dictfile_name
        print("Dict saved and cleaned, now tmp is: ",tmp)
        
print(tmp)
with open(path+'\\'+tmp+'.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
dictionary={}
print("All dict saved")