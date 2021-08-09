"""
Purposes of this code is to create pickle label for every video inside ../examples/demo/test/
"""
import glob
import pickle

path = 'C:\\Users\\rashi\\Documents\\FallDataset\\FallDataset_2\\le2i_annotated\\trainData\\trainData.pickle'
with open(path, 'rb') as handle:
    annotations = pickle.load(handle)
print(annotations)