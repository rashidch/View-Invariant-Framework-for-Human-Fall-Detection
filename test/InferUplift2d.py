from test.uplift2d import inferencealphaposeto3D
from test.uplift2d import save_to_json
import os

path='source3d/json_test/taoyuan_angle2.json'
base= os.path.splitext(os.path.basename(path))[0]
inference_result=inferencealphaposeto3D(path,fixing=False,save_npy=True)

