import os 
import sys
import torch 
import numpy as np 
import json

need_pytorch3d = False
try:
    import pytorch3d
except:
    need_pytorch3d = True 

from pytorch3d.io import load_obj
mesh_path = "data/meshes/Teeth_Model_1.obj"
verts, faces, properties = load_obj(mesh_path)
print(verts)
