import os
import sys
import torch

import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
def setup_device():
    # Set up device (GPU if available, else CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device

def main():
    
    device = setup_device()
    # Load the obj and ignore the textures and materials.
    verts, faces_idx, _ = load_obj("./data/meshes/teapot.obj")
    faces = faces_idx.verts_idx

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    teapot_mesh = Meshes(
        verts=[verts.to(device)],   
        faces=[faces.to(device)], 
        textures=textures
    ) 

    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
    # edges. Refer to blending.py for more details. 
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=256, 
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=100, 
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )


    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=256, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    # We can add a point light in front of the object. 
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )

    # Select the viewpoint using spherical angles  
    distance = 3   # distance from camera to the object
    elevation = 50.0   # angle of elevation in degrees
    azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

    # Render the teapot providing the values of R and T. 
    silhouette = silhouette_renderer(meshes_world=teapot_mesh, R=R, T=T)
    image_ref = phong_renderer(meshes_world=teapot_mesh, R=R, T=T)

    silhouette = silhouette.cpu().numpy()
    image_ref = image_ref.cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silhouette.squeeze()[..., 3])  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.imshow(image_ref.squeeze())
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    main()