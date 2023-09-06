import os
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    look_at_view_transform,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    HardFlatShader,
    TexturesVertex,
    HardPhongShader,
    Textures

)

from pytorch3d.renderer.mesh.shader import SimpleColorShader

from helpers.visualization.img_grid import image_grid
import matplotlib.pyplot as plt

def download_cow_mesh(data_dir):
    # Download the cow mesh files if they don't exist
    cow_mesh_files = [
        os.path.join(data_dir, fl) for fl in ("cow.obj", "cow.mtl", "cow_texture.png")
    ]
    if any(not os.path.isfile(f) for f in cow_mesh_files):
        os.makedirs(data_dir, exist_ok=True)
        os.system(f"wget -P {data_dir} " + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj")
        os.system(f"wget -P {data_dir} " + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl")
        os.system(f"wget -P {data_dir} " + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png")

def setup_device():
    # Set up device (GPU if available, else CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device

def load_normalize_mesh(data_dir, device):
    # Load cow mesh, normalize and center
    obj_filename = os.path.join(data_dir, "cow.obj")
    mesh = load_objs_as_meshes([obj_filename], device=device)
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))
    # Initialize each vertex to be white in color.
    # verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    # mesh.textures = TexturesVertex(verts_features=verts_rgb.to(device))
    return mesh

def generate_cameras_viewpoints(num_views, device):
    # Get a batch of viewing angles. 
    elev = torch.linspace(0, -90, num_views)
    azim = torch.linspace(0, -180, num_views)


    # Initialize an OpenGL perspective camera that represents a batch of different 
    # viewing angles. All the cameras helper methods support mixed type inputs and 
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and 
    # then specify elevation and azimuth angles for each viewpoint as tensors. 
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras, R, T
    

def setup_renderer_and_lights(camera, device):
    # Place a point light in front of the object. As mentioned above, the front of 
    # the cow is facing the -z direction. 
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])



    # Define the settings for rasterization and shading. Here we set the output 
    # image to be of size 128X128. As we are rendering images for visualization 
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to 
    # rasterize_meshes.py for explanations of these parameters.  We also leave 
    # bin_size and max_faces_per_bin to their default values of None, which sets 
    # their values using heuristics and ensures that the faster coarse-to-fine 
    # rasterization method is used.  Refer to docs/notes/renderer.md for an 
    # explanation of the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=128, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured 
    # Phong shader will interpolate the texture uv coordinates for each vertex, 
    # sample from a texture image and apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
        device=device, 
        cameras=camera,
        lights=lights
    )
    )

    return renderer, lights


def main():

    data_dir = "data/meshes/cow_mesh/"
    device = setup_device()
    mesh = load_normalize_mesh(data_dir, device)
    
    # the number of different viewpoints from which we want to render the mesh.
    num_views = 9
    cameras, R, T = generate_cameras_viewpoints(num_views, device)
        
    # We arbitrarily choose one particular view that will be used to visualize 
    # results
    camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], 
                                    T=T[None, 1, ...]) 

    
    renderer, lights = setup_renderer_and_lights(camera, device)
    # Create a batch of meshes by repeating the cow mesh and associated textures. 
    # Meshes has a useful `extend` method which allows us do this very easily. 
    # This also extends the textures. 
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)

    # Our multi-view cow dataset will be represented by these 2 lists of tensors,
    # each of length num_views.
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    target_cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], 
                                            T=T[None, i, ...]) for i in range(num_views)]
    
    # RGB images
    image_grid(target_images.cpu().numpy(), rows=3, cols=3, rgb=True)
    plt.show()


if __name__ == "__main__":
    main()