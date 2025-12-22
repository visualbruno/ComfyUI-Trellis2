import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageSequence, ImageOps
from pathlib import Path
import numpy as np
import json
import trimesh as Trimesh
from tqdm import tqdm

import folder_paths
import node_helpers
import hashlib
import cv2
import gc

import cumesh as CuMesh

import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmeshpy

import nvdiffrast.torch as dr
from flex_gemm.ops.grid_sample import grid_sample_3d

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.utils

from .trellis2.pipelines import Trellis2ImageTo3DPipeline

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
    
def tensor2pil(image: torch.Tensor) -> Image.Image:
    """
    Accepts either:
      - (H,W,C)
      - (1,H,W,C)
    Returns a PIL RGB/RGBA image depending on channels.
    """
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 4:
            # Expect (B,H,W,C); allow only B==1 here
            if t.shape[0] != 1:
                raise ValueError(f"tensor2pil expects batch of 1, got batch={t.shape[0]}")
            t = t[0]
        elif t.ndim != 3:
            raise ValueError(f"tensor2pil expects (H,W,C) or (1,H,W,C), got shape={tuple(t.shape)}")

        arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    raise TypeError(f"tensor2pil expected torch.Tensor, got {type(image)}")

def tensor_batch_to_pil_list(images: torch.Tensor, max_views: int = 4) -> list[Image.Image]:
    """
    Converts a ComfyUI IMAGE tensor (B,H,W,C) into a list of PIL images.
    Caps to max_views for safety.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for IMAGE, got {type(images)}")

    if images.ndim == 4:
        b = int(images.shape[0])
        n = min(b, int(max_views))
        return [tensor2pil(images[i:i+1]) for i in range(n)]

    if images.ndim == 3:
        return [tensor2pil(images)]

    raise ValueError(f"Unsupported IMAGE tensor shape: {tuple(images.shape)}")

def convert_tensor_images_to_pil(images):
    pil_array = []
    
    for image in images:
        pil_array.append(tensor2pil(image))
        
    return pil_array
    
def simplify_with_meshlib(vertices, faces, target=1000000):
    current_faces_num = len(faces)
    print(f'Current Faces Number: {current_faces_num}')
    
    if current_faces_num<target:
        return

    settings = mrmeshpy.DecimateSettings()
    faces_to_delete = current_faces_num - target
    settings.maxDeletedFaces = faces_to_delete                        
    settings.packMesh = True
    
    print('Generating Meshlib Mesh ...')
    mesh = mrmeshnumpy.meshFromFacesVerts(faces, vertices)
    print('Packing Optimally ...')
    mesh.packOptimally()
    print('Decimating ...')
    mrmeshpy.decimateMesh(mesh, settings)
    
    new_vertices = mrmeshnumpy.getNumpyVerts(mesh)
    new_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)               
    
    print(f"Reduced faces, resulting in {len(new_vertices)} vertices and {len(new_faces)} faces")
        
    return new_vertices, new_faces

class Trellis2LoadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelname": (["TRELLIS.2-4B"],),
                "backend": (["flash_attn","xformers"],{"default":"xformers"}),
                "device": (["cpu","cuda"],{"default":"cuda"}),
                "low_vram": ("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("TRELLIS2PIPELINE", )
    RETURN_NAMES = ("pipeline", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, modelname, backend, device, low_vram):
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
        os.environ['ATTN_BACKEND'] = backend
        
        download_path = os.path.join(folder_paths.models_dir,"microsoft")
        model_path = os.path.join(download_path, modelname)
        
        hf_model_name = f"microsoft/{modelname}"
        
        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=hf_model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
            
        dinov3_model_path = os.path.join(folder_paths.models_dir,"facebook","dinov3-vitl16-pretrain-lvd1689m","model.safetensors")
        if not os.path.exists(dinov3_model_path):
            raise Exception("Facebook Dinov3 model not found in models/facebook/dinov3-vitl16-pretrain-lvd1689m folder")
        
        trellis_image_large_path = os.path.join(folder_paths.models_dir,"microsoft","Trellis-Image-Large","ckpts","ss_dec_conv3d_16l8_fp16.safetensors")
        if not os.path.exists(trellis_image_large_path):
            print('Trellis-Image-Large ss_dec_conv3d_16l8_fp16 files not found. Trying to download the files from huggingface ...')
            import requests
            url = "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.json?download=true"
            filename = os.path.join(folder_paths.models_dir,"microsoft","Trellis-Image-Large","ckpts","ss_dec_conv3d_16l8_fp16.json")
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print("Download ss_dec_conv3d_16l8_fp16.json complete!")
            else:
                raise Exception("Cannot download Trellis-Image-Large file ss_dec_conv3d_16l8_fp16.json")
            
            url = "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.safetensors?download=true"
            filename = os.path.join(folder_paths.models_dir,"microsoft","Trellis-Image-Large","ckpts","ss_dec_conv3d_16l8_fp16.safetensors")

            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print("Download ss_dec_conv3d_16l8_fp16.safetensors complete!")
            else:
                raise Exception("Cannot download Trellis-Image-Large file ss_dec_conv3d_16l8_fp16.safetensors")
        
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        pipeline.low_vram = low_vram
        
        if device=="cuda":
            if low_vram:
                pipeline.cuda()
            else:
                pipeline.to(device)
        else:
            pipeline.to(device)
        
        return (pipeline,)
        
class Trellis2MeshWithVoxelGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "pipeline_type": (["512","1024","1024_cascade","1536_cascade"],{"default":"1024_cascade"}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "max_num_tokens": ("INT",{"default":49152,"min":0,"max":999999}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, seed, pipeline_type, sparse_structure_steps, shape_steps, texture_steps, max_num_tokens, max_views):
        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images
        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps}
        shape_slat_sampler_params = {"steps":shape_steps}
        tex_slat_sampler_params = {"steps":texture_steps}
        
        mesh = pipeline.run(image=image_in, seed=seed, pipeline_type=pipeline_type,
                            sparse_structure_sampler_params=sparse_structure_sampler_params,
                            shape_slat_sampler_params=shape_slat_sampler_params,
                            tex_slat_sampler_params=tex_slat_sampler_params,
                            max_num_tokens=max_num_tokens)[0]
  
        return (mesh,)    

class Trellis2LoadImageWithTransparency:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "Trellis2Wrapper"

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", )
    RETURN_NAMES = ("image", "mask", "image_with_alpha")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        output_images_ori = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            
            output_images_ori.append(pil2tensor(i))

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
            output_image_ori = torch.cat(output_images_ori, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
            output_image_ori = output_images_ori[0]

        return (output_image, output_mask, output_image_ori)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True  

class Trellis2SimplifyMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "target_face_num": ("INT",{"default":1000000,"min":1,"max":16000000}),
                "method": (["Cumesh","Meshlib"],{"default":"Cumesh"}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, target_face_num, method):        
        if method=="Cumesh":
            mesh.simplify_with_cumesh(target = target_face_num)
        elif method=="Meshlib":
            mesh.simplify_with_meshlib(target = target_face_num)
        else:
            raise Exception("Unknown simplification method")             
        
        return (mesh,)          
        
class Trellis2MeshWithVoxelToTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh):       
        vertices_np = mesh.vertices.cpu().numpy()
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]        
        
        trimesh = Trimesh.Trimesh(
            vertices=vertices_np,
            faces=mesh.faces.cpu().numpy(),
            process=False
        )
        
        return (trimesh,)
        
class Trellis2ExportMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/Trellis2"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "save_file": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, filename_prefix, file_format, save_file=True):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(exist_ok=True)
        if save_file:
            trimesh.export(output_glb_path, file_type=file_format)
            relative_path = Path(subfolder) / f'{filename}_{counter:05}_.{file_format}'
        else:
            temp_file = Path(full_output_folder, f'hy3dtemp_.{file_format}')
            trimesh.export(temp_file, file_type=file_format)
            relative_path = Path(subfolder) / f'hy3dtemp_.{file_format}'
        
        return (str(relative_path), )        
        
class Trellis2PostProcessMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "fill_holes": ("BOOLEAN", {"default":True}),
                "fill_holes_max_perimeter": ("FLOAT",{"default":0.03,"min":0.001,"max":99.999,"step":0.001}),
                "remove_duplicate_faces": ("BOOLEAN",{"default":True}),
                "repair_non_manifold_edges": ("BOOLEAN", {"default":True}),
                "remove_non_manifold_faces": ("BOOLEAN", {"default":True}),
                "remove_small_connected_components": ("BOOLEAN", {"default":True}),
                "remove_small_connected_components_size": ("FLOAT", {"default":0.00001,"min":0.00001,"max":9.99999,"step":0.00001}),                
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, fill_holes, fill_holes_max_perimeter, remove_duplicate_faces, repair_non_manifold_edges, remove_non_manifold_faces, remove_small_connected_components, remove_small_connected_components_size):
        vertices = mesh.vertices
        faces = mesh.faces

        # Move data to GPU
        vertices = vertices.cuda()
        faces = faces.cuda()
        
        # Initialize CUDA mesh handler
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        print(f"Current vertices: {cumesh.num_vertices}, faces: {cumesh.num_faces}")
        
        if fill_holes:
            cumesh.fill_holes(max_hole_perimeter=fill_holes_max_perimeter)
            print(f"After filling holes: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
            
        if remove_duplicate_faces:
            print('Removing duplicate faces ...')
            cumesh.remove_duplicate_faces()
            
        if repair_non_manifold_edges:
            print('Repairing non manifold edges ...')
            cumesh.repair_non_manifold_edges()
            
        if remove_non_manifold_faces:
            print('Removing non manifold faces ...')
            cumesh.remove_non_manifold_faces()
            
        if remove_small_connected_components:
            print('Removing small connected components ...')
            cumesh.remove_small_connected_components(remove_small_connected_components_size)        
        
        print(f"After initial cleanup: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")                                   
        
        cumesh.unify_face_orientations()        
        
        new_vertices, new_faces = cumesh.read()
        
        mesh.vertices = new_vertices.to(mesh.device)
        mesh.faces = new_faces.to(mesh.device) 
        
        del cumesh
        gc.collect()          
                
        return (mesh,)
       
class Trellis2UnWrapAndRasterizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":90.0,"min":0.0,"max":359.9}),
                "mesh_cluster_refine_iterations": ("INT",{"default":0}),
                "mesh_cluster_global_iterations": ("INT",{"default":1}),
                "mesh_cluster_smooth_strength": ("INT",{"default":1}),                
                "texture_size": ("INT",{"default":1024, "min":512, "max":16384}),
                "texture_alpha_mode": (["OPAQUE","MASK","BLEND"],{"default":"OPAQUE"}),
                "double_side_material": ("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, mesh_cluster_threshold_cone_half_angle_rad, mesh_cluster_refine_iterations, mesh_cluster_global_iterations, mesh_cluster_smooth_strength, texture_size, texture_alpha_mode, double_side_material):
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        
        vertices = mesh.vertices
        faces = mesh.faces
        attr_volume = mesh.attrs
        coords = mesh.coords
        attr_layout = mesh.layout
        voxel_size = mesh.voxel_size  
        
        mesh_cluster_threshold_cone_half_angle_rad = np.radians(mesh_cluster_threshold_cone_half_angle_rad)

        # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)

        # Calculate grid dimensions based on AABB and voxel size                
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = [voxel_size, voxel_size, voxel_size]
            if isinstance(voxel_size, (list, tuple)):
                voxel_size = np.array(voxel_size)
            if isinstance(voxel_size, np.ndarray):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            if isinstance(grid_size, int):
                grid_size = [grid_size, grid_size, grid_size]
            if isinstance(grid_size, (list, tuple)):
                grid_size = np.array(grid_size)
            if isinstance(grid_size, np.ndarray):
                grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
            voxel_size = (aabb[1] - aabb[0]) / grid_size       
        
            print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")        
        
        vertices = vertices.cuda()
        faces = faces.cuda()        
        
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        
        # Build BVH for the current mesh to guide remeshing
        print(f"Building BVH for current mesh...")
        bvh = CuMesh.cuBVH(vertices, faces)        
        
        print('Unwrapping ...')        
        out_vertices, out_faces, out_uvs, out_vmaps = cumesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
                "refine_iterations": mesh_cluster_refine_iterations,
                "global_iterations": mesh_cluster_global_iterations,
                "smooth_strength": mesh_cluster_smooth_strength,
            },
            return_vmaps=True,
            verbose=True,
        )
        
        out_vertices = out_vertices.cuda()
        out_faces = out_faces.cuda()
        out_uvs = out_uvs.cuda()
        out_vmaps = out_vmaps.cuda()
        cumesh.compute_vertex_normals()
        out_normals = cumesh.read_vertex_normals()[out_vmaps]        

        print("Sampling attributes...")
        # Setup differentiable rasterizer context
        ctx = dr.RasterizeCudaContext()
        # Prepare UV coordinates for rasterization (rendering in UV space)
        uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
        rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
        
        # Rasterize in chunks to save memory
        for i in range(0, out_faces.shape[0], 100000):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_rast, out_faces[i:i+100000],
                resolution=[texture_size, texture_size],
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i # Store face ID in alpha channel
            rast = torch.where(mask_chunk, rast_chunk, rast)
        
        # Mask of valid pixels in texture
        mask = rast[0, ..., 3] > 0
        
        # Interpolate 3D positions in UV space (finding 3D coord for every texel)
        pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
        valid_pos = pos[mask]
        
        # Map these positions back to the *original* high-res mesh to get accurate attributes
        # This corrects geometric errors introduced by simplification/remeshing
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
        
        # Trilinear sampling from the attribute volume (Color, Material props)
        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )
        
        # --- Texture Post-Processing & Material Construction ---
        print("Finalizing mesh...")
        
        mask = mask.cpu().numpy()
        
        # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
        base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha_mode = texture_alpha_mode
        
        # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
        mask_inv = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        
        # Create PBR material
        # Standard PBR packs Metallic and Roughness into Blue and Green channels
        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=alpha_mode,
            doubleSided=double_side_material,
        )        
        
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()
        uvs_np = out_uvs.cpu().numpy()
        normals_np = out_normals.cpu().numpy()
        
        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
        
        textured_mesh = Trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=Trimesh.visual.TextureVisuals(uv=uvs_np,material=material)
        )   

        del cumesh
        gc.collect()          
                
        return (textured_mesh,)
        
class Trellis2MeshWithVoxelAdvancedGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "pipeline_type": (["512","1024","1024_cascade","1536_cascade"],{"default":"1024_cascade"}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":7.5}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.7}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":3.0}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":7.5}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.7}),
                "shape_rescale_t": ("FLOAT",{"default":3.0}),                
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":7.0}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.7}),
                "texture_rescale_t": ("FLOAT",{"default":3.0}),                
                "max_num_tokens": ("INT",{"default":49152,"min":0,"max":999999}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, max_views, seed, pipeline_type, sparse_structure_steps, 
        sparse_structure_guidance_strength, 
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,        
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale,
        texture_rescale_t,        
        max_num_tokens):

        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images
        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"rescale_t":sparse_structure_rescale_t}        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"rescale_t":shape_rescale_t}       
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"rescale_t":texture_rescale_t}
                
        mesh = pipeline.run(
            image=image_in,
            seed=seed,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            shape_slat_sampler_params=shape_slat_sampler_params,
            tex_slat_sampler_params=tex_slat_sampler_params,
            max_num_tokens=max_num_tokens
        )[0]

        return (mesh,)    

class Trellis2PostProcessAndUnWrapAndRasterizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":90.0,"min":0.0,"max":359.9}),
                "mesh_cluster_refine_iterations": ("INT",{"default":0}),
                "mesh_cluster_global_iterations": ("INT",{"default":1}),
                "mesh_cluster_smooth_strength": ("INT",{"default":1}),                
                "texture_size": ("INT",{"default":1024, "min":512, "max":16384}),
                "remesh": ("BOOLEAN",{"default":True}),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "remesh_project": ("FLOAT",{"default":0.0}),
                "target_face_num": ("INT",{"default":1000000,"min":1,"max":16000000}),
                "simplify_method": (["Cumesh","Meshlib"],{"default":"Cumesh"}),
                "fill_holes": ("BOOLEAN", {"default":True}),
                "fill_holes_max_perimeter": ("FLOAT",{"default":0.03,"min":0.001,"max":99.999,"step":0.001}),
                "texture_alpha_mode": (["OPAQUE","MASK","BLEND"],{"default":"OPAQUE"}),
                "dual_contouring_resolution": (["Auto","128","256","512","1024","2048"],{"default":"Auto"}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, mesh_cluster_threshold_cone_half_angle_rad, mesh_cluster_refine_iterations, mesh_cluster_global_iterations, mesh_cluster_smooth_strength, texture_size, remesh, remesh_band, remesh_project, target_face_num, simplify_method, fill_holes, fill_holes_max_perimeter, texture_alpha_mode, dual_contouring_resolution):
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        
        vertices = mesh.vertices
        faces = mesh.faces
        attr_volume = mesh.attrs
        coords = mesh.coords
        attr_layout = mesh.layout
        voxel_size = mesh.voxel_size  
        
        mesh_cluster_threshold_cone_half_angle_rad = np.radians(mesh_cluster_threshold_cone_half_angle_rad)

        # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)

        # Calculate grid dimensions based on AABB and voxel size                
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = [voxel_size, voxel_size, voxel_size]
            if isinstance(voxel_size, (list, tuple)):
                voxel_size = np.array(voxel_size)
            if isinstance(voxel_size, np.ndarray):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            if isinstance(grid_size, int):
                grid_size = [grid_size, grid_size, grid_size]
            if isinstance(grid_size, (list, tuple)):
                grid_size = np.array(grid_size)
            if isinstance(grid_size, np.ndarray):
                grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
            voxel_size = (aabb[1] - aabb[0]) / grid_size       
        
            print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")        
        
        vertices = vertices.cuda()
        faces = faces.cuda()                
        
        # Initialize CUDA mesh handler
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        print(f"Current vertices: {cumesh.num_vertices}, faces: {cumesh.num_faces}")
        
        # --- Initial Mesh Cleaning ---
        # Fills holes as much as we can before processing
        if fill_holes:
            cumesh.fill_holes(max_hole_perimeter=fill_holes_max_perimeter)
            print(f"After filling holes: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")        
            
        # Build BVH for the current mesh to guide remeshing
        print(f"Building BVH for current mesh...")
        bvh = CuMesh.cuBVH(vertices, faces)
            
        print("Cleaning mesh...")        
        # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
        if not remesh:            
            # Step 1: Clean up topology (duplicates, non-manifolds, isolated parts)
            cumesh.remove_duplicate_faces()
            cumesh.repair_non_manifold_edges()
            cumesh.remove_small_connected_components(1e-5)
            
            if fill_holes:
                cumesh.fill_holes(max_hole_perimeter=fill_holes_max_perimeter)
            
            print(f"After initial cleanup: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")                            
                
            # Step 2: Unify face orientations
            cumesh.unify_face_orientations()
        
        # --- Branch 2: Remeshing Pipeline ---
        else:
            center = aabb.mean(dim=0)
            scale = (aabb[1] - aabb[0]).max().item()
            
            if dual_contouring_resolution == "Auto":
                resolution = grid_size.max().item()
                print(f"Dual Contouring resolution: {resolution}")
            else:
                resolution = int(dual_contouring_resolution)
            
            # Perform Dual Contouring remeshing (rebuilds topology)
            cumesh.init(*CuMesh.remeshing.remesh_narrow_band_dc(
                vertices, faces,
                center = center,
                scale = (resolution + 3 * remesh_band) / resolution * scale,
                resolution = resolution,
                band = remesh_band,
                project_back = remesh_project, # Snaps vertices back to original surface
                verbose = True,
                bvh = bvh,
            ))
            
            print(f"After remeshing: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
            
            # Step 2: Unify face orientations
            cumesh.unify_face_orientations()            

        if simplify_method == 'Cumesh':
            cumesh.simplify(target_face_num, verbose=True)
        elif simplify_method == 'Meshlib':
             # GPU -> CPU -> Meshlib -> CPU -> GPU
            v, f = cumesh.read()
            new_vertices, new_faces = simplify_with_meshlib(v.cpu().numpy(), f.cpu().numpy(), target_face_num)
            cumesh.init(torch.from_numpy(new_vertices).float().cuda(), torch.from_numpy(new_faces).int().cuda())

        print(f"After simplifying: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")            
        
        print('Unwrapping ...')        
        out_vertices, out_faces, out_uvs, out_vmaps = cumesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
                "refine_iterations": mesh_cluster_refine_iterations,
                "global_iterations": mesh_cluster_global_iterations,
                "smooth_strength": mesh_cluster_smooth_strength,
            },
            return_vmaps=True,
            verbose=True,
        )
        
        out_vertices = out_vertices.cuda()
        out_faces = out_faces.cuda()
        out_uvs = out_uvs.cuda()
        out_vmaps = out_vmaps.cuda()
        cumesh.compute_vertex_normals()
        out_normals = cumesh.read_vertex_normals()[out_vmaps]        

        print("Sampling attributes...")
        # Setup differentiable rasterizer context
        ctx = dr.RasterizeCudaContext()
        # Prepare UV coordinates for rasterization (rendering in UV space)
        uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
        rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
        
        # Rasterize in chunks to save memory
        for i in range(0, out_faces.shape[0], 100000):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_rast, out_faces[i:i+100000],
                resolution=[texture_size, texture_size],
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i # Store face ID in alpha channel
            rast = torch.where(mask_chunk, rast_chunk, rast)
        
        # Mask of valid pixels in texture
        mask = rast[0, ..., 3] > 0
        
        # Interpolate 3D positions in UV space (finding 3D coord for every texel)
        pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
        valid_pos = pos[mask]
        
        # Map these positions back to the *original* high-res mesh to get accurate attributes
        # This corrects geometric errors introduced by simplification/remeshing
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
        
        # Trilinear sampling from the attribute volume (Color, Material props)
        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )
        
        # --- Texture Post-Processing & Material Construction ---
        print("Finalizing mesh...")
        
        mask = mask.cpu().numpy()
        
        # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
        base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha_mode = texture_alpha_mode
        
        # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
        mask_inv = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        
        # Create PBR material
        # Standard PBR packs Metallic and Roughness into Blue and Green channels
        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=alpha_mode,
            doubleSided=True if not remesh else False,
        )        
        
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()
        uvs_np = out_uvs.cpu().numpy()
        normals_np = out_normals.cpu().numpy()
        
        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
        
        textured_mesh = Trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=Trimesh.visual.TextureVisuals(uv=uvs_np,material=material)
        )        
        
        del cumesh
        gc.collect()          
        
        return (textured_mesh,)    

class Trellis2Remesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "remesh_project": ("FLOAT",{"default":0.0}),
                "fill_holes": ("BOOLEAN", {"default":True}),
                "fill_holes_max_perimeter": ("FLOAT",{"default":0.03,"min":0.001,"max":99.999,"step":0.001}),
                "dual_contouring_resolution": (["Auto","128","256","512","1024","2048"],{"default":"Auto"}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, remesh_band, remesh_project, fill_holes, fill_holes_max_perimeter, dual_contouring_resolution):
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        
        vertices = mesh.vertices
        faces = mesh.faces
        attr_volume = mesh.attrs
        coords = mesh.coords
        attr_layout = mesh.layout
        voxel_size = mesh.voxel_size        
        
        # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)

        # Calculate grid dimensions based on AABB and voxel size                
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = [voxel_size, voxel_size, voxel_size]
            if isinstance(voxel_size, (list, tuple)):
                voxel_size = np.array(voxel_size)
            if isinstance(voxel_size, np.ndarray):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            if isinstance(grid_size, int):
                grid_size = [grid_size, grid_size, grid_size]
            if isinstance(grid_size, (list, tuple)):
                grid_size = np.array(grid_size)
            if isinstance(grid_size, np.ndarray):
                grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
            voxel_size = (aabb[1] - aabb[0]) / grid_size

        # Move data to GPU
        vertices = vertices.cuda()
        faces = faces.cuda()
        
        # Initialize CUDA mesh handler
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        print(f"Current vertices: {cumesh.num_vertices}, faces: {cumesh.num_faces}")
        
        # --- Initial Mesh Cleaning ---
        # Fills holes as much as we can before processing
        if fill_holes:
            cumesh.fill_holes(max_hole_perimeter=fill_holes_max_perimeter)
            print(f"After filling holes: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
        
        vertices, faces = cumesh.read()
            
        # Build BVH for the current mesh to guide remeshing
        print(f"Building BVH for current mesh...")
        bvh = CuMesh.cuBVH(vertices, faces)
            
        print("Cleaning mesh...")        
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        
        if dual_contouring_resolution == "Auto":
            resolution = grid_size.max().item()
            print(f"Dual Contouring resolution: {resolution}")
        else:
            resolution = int(dual_contouring_resolution)
        
        # Perform Dual Contouring remeshing (rebuilds topology)
        cumesh.init(*CuMesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = True,
            bvh = bvh,
        ))
        
        print(f"After remeshing: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")                          
        
        # Step 2: Unify face orientations
        cumesh.unify_face_orientations()        
        
        new_vertices, new_faces = cumesh.read()
        
        mesh.vertices = new_vertices.to(mesh.device)
        mesh.faces = new_faces.to(mesh.device) 
        
        del cumesh
        gc.collect()          
                
        return (mesh,)        

NODE_CLASS_MAPPINGS = {
    "Trellis2LoadModel": Trellis2LoadModel,
    "Trellis2MeshWithVoxelGenerator": Trellis2MeshWithVoxelGenerator,
    "Trellis2LoadImageWithTransparency": Trellis2LoadImageWithTransparency,
    "Trellis2SimplifyMesh": Trellis2SimplifyMesh,
    "Trellis2MeshWithVoxelToTrimesh": Trellis2MeshWithVoxelToTrimesh,
    "Trellis2ExportMesh": Trellis2ExportMesh,
    "Trellis2PostProcessMesh": Trellis2PostProcessMesh,
    "Trellis2UnWrapAndRasterizer": Trellis2UnWrapAndRasterizer,
    "Trellis2MeshWithVoxelAdvancedGenerator": Trellis2MeshWithVoxelAdvancedGenerator,
    "Trellis2PostProcessAndUnWrapAndRasterizer": Trellis2PostProcessAndUnWrapAndRasterizer,
    "Trellis2Remesh": Trellis2Remesh,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2LoadModel": "Trellis2 - LoadModel",
    "Trellis2MeshWithVoxelGenerator": "Trellis2 - Mesh With Voxel Generator",
    "Trellis2LoadImageWithTransparency": "Trellis2 - Load Image with Transparency",
    "Trellis2SimplifyMesh": "Trellis2 - Simplify Mesh",
    "Trellis2MeshWithVoxelToTrimesh": "Trellis2 - Mesh With Voxel To Trimesh",
    "Trellis2ExportMesh": "Trellis2 - Export Mesh",
    "Trellis2PostProcessMesh": "Trellis2 - PostProcess Mesh",
    "Trellis2UnWrapAndRasterizer": "Trellis2 - UV Unwrap and Rasterize",
    "Trellis2MeshWithVoxelAdvancedGenerator": "Trellis2 - Mesh With Voxel Advanced Generator",
    "Trellis2PostProcessAndUnWrapAndRasterizer": "Trellis2 - Post Process/UnWrap and Rasterize",
    "Trellis2Remesh": "Trellis2 - Remesh",
    }