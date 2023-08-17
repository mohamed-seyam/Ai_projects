import os 
import sys
import torch 
import numpy as np 
import json
from pathlib import Path
from pytorch3d.io import load_obj
from pytorch3d.io import IO 
from pytorch3d.structures.meshes import Meshes
from pytorch3d.io.obj_io import load_obj, load_objs_as_meshes, save_obj
from pytorch3d.io.ply_io import load_ply
from pytorch3d.utils import ico_sphere

from pytorch3d.ops import sample_points_from_meshes

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
def _load_mesh(mesh_path: str , device = "cpu")-> Meshes:
    extension = mesh_path.split(".")[-1]
    assert extension in ["obj", "stl", "off"]
    mesh = IO().load_mesh(mesh_path, device=device)
    return mesh 

def point_cloud_to_mesh(pcl, output_mesh_name):
    #TODO didn't tested yet
    extension = output_mesh_name.split(".")[-1] 
    assert extension == "ply"
    IO().save_pointcloud(pcl, output_mesh_name)


def _load_obj(object_filepath, device = "cpu"):
    """read an object file to know more 
    https://pytorch3d.org/docs/meshes_io
    """
    extension = object_filepath.split(".")[-1]
    assert extension == "obj"
    verts, faces, aux = load_obj(object_filepath, device)

    return verts, faces, aux 

def _load_ply(ply_filepath, device = "cpu"):
    """read an object file to know more 
    https://pytorch3d.org/docs/meshes_io
    """
    extension = ply_filepath.split(".")[-1]
    assert extension == "ply"
    verts, faces = load_ply(ply_filepath, device)

    return verts, faces 

def _load_objs_as_meshes(folder_path):
    obj_paths = [str(p) for p in Path(folder_path).glob("*.obj")]
    meshes_batch = load_objs_as_meshes(obj_paths)
    return meshes_batch

def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

def main():
    target_mesh_path = "data/meshes/dolphin.obj"
    device = torch.device("cpu")

    verts, faces, aux = _load_obj(target_mesh_path)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    # We construct a Meshes structure for the target mesh
    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])
    src_mesh = ico_sphere(4, device)
    IO().save_mesh(src_mesh, "src_mesh.obj")
    # We will learn to deform the source mesh by offsetting its vertices
    # The shape of the deform parameters is equal to the total number of vertices in src_mesh
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

    # The optimizer
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)   

    # Number of optimization steps
    Niter = 2000
    # Weight for the chamfer loss
    w_chamfer = 1.0 
    # Weight for mesh edge loss
    w_edge = 1.0 
    # Weight for mesh normal consistency
    w_normal = 0.01 
    # Weight for mesh laplacian smoothing
    w_laplacian = 0.1 
    # Plot period for the losses
    plot_period = 10
    loop = tqdm(range(Niter))

    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []



    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        
        # We sample 5k points from the surface of each mesh 
        sample_trg = sample_points_from_meshes(trg_mesh, 5000)
        sample_src = sample_points_from_meshes(new_src_mesh, 5000)
        
        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
        
        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(new_src_mesh)
        
        # mesh normal consistency
        loss_normal = mesh_normal_consistency(new_src_mesh)
        
        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        
        # Weighted sum of the losses
        loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
        
        # Print the losses
        loop.set_description('total_loss = %.6f' % loss)
        
        # Save the losses for plotting
        chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))
        
        # Plot mesh
        if i % plot_period == 0:
            IO().save_mesh(new_src_mesh, f"data/tmp/new_src_mesh_{i}.obj")
            
        # Optimization step
        loss.backward()
        optimizer.step()
if __name__ == "__main__":
    main()