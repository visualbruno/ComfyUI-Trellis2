import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageSequence, ImageOps
from pathlib import Path
import numpy as np
import json
import trimesh as Trimesh
from tqdm import tqdm
import time
import shutil
import uuid
import triton
import triton.compiler
import math

import folder_paths
import node_helpers
import hashlib
import cv2
import gc
import copy

import pymeshlab

import cumesh as CuMesh
import o_voxel

import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmeshpy

import nvdiffrast.torch as dr
from flex_gemm.ops.grid_sample import grid_sample_3d

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.utils

from .trellis2.pipelines import Trellis2ImageTo3DPipeline
from .trellis2.representations import Mesh, MeshWithVoxel
from .trellis2.modules.attention import config
from .trellis2.modules.sparse import config as sparseconfig
from .trellis2.pipelines import samplers
from .trellis2.modules.sparse import SparseTensor

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

BASE_CACHE_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "triton_caches"
#os.environ["TRITON_ALWAYS_COMPILE"] = "1"
#os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"]="1"      

to_pil = transforms.ToPILImage()

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False

any = AnyType("*")

class ViewConfig:
    def __init__(self,image,azimuth,elevation,prompt):
        self.image = image
        self.azimuth = azimuth
        self.elevation = elevation
        self.prompt = prompt

def inpaint_channel(channel, mask_inv, radius=1, mode = cv2.INPAINT_TELEA):
    out = cv2.inpaint(channel, mask_inv, radius, mode)
    if out.ndim == 2:
        out = out[..., None]
    return out

def rotate_triton_cache():
    """
    Creates a new cache directory and attempts to clean up old ones.
    """
    # 1. Create the base directory if it doesn't exist
    BASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Generate a unique ID for this specific run
    run_id = f"cache_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    new_cache_path = BASE_CACHE_DIR / run_id
    new_cache_path.mkdir()

    # 3. Point Triton to this NEW empty folder
    # This forces a recompile without needing to delete the locked file immediately
    os.environ["TRITON_CACHE_DIR"] = str(new_cache_path)
    print(f"[TrellisNode] 🔄 Switched to fresh Triton cache: {new_cache_path.name}")

    # 4. Garbage Collection: Try to delete OLD cache folders
    # We wrap this in a try/except so if Windows locks a file, we just skip it
    # and leave it for the next cleanup cycle.
    cleanup_old_caches(current_active=new_cache_path)                

def cleanup_old_caches(current_active):
    """
    Iterates through the cache folder and deletes anything that isn't the current active one.
    If a file is locked by Windows, it silently fails and leaves it for later.
    """
    for item in BASE_CACHE_DIR.iterdir():
        if item.is_dir() and item != current_active:
            try:
                shutil.rmtree(item)
                print(f"[TrellisNode] 🧹 Cleaned up old cache: {item.name}")
            except OSError:
                # This is expected on Windows! The file is locked.
                # We just ignore it and try again next time the node runs.
                pass 

def parse_string_to_int_list(number_string):
  """
  Parses a string containing comma-separated numbers into a list of integers.

  Args:
    number_string: A string containing comma-separated numbers (e.g., "20000,10000,5000").

  Returns:
    A list of integers parsed from the input string.
    Returns an empty list if the input string is empty or None.
  """
  if not number_string:
    return []

  try:
    # Split the string by comma and convert each part to an integer
    int_list = [int(num.strip()) for num in number_string.split(',')]
    return int_list
  except ValueError as e:
    print(f"Error converting string to integer: {e}. Please ensure all values are valid numbers.")
    return []

def reset_cuda():    
    # Synchronize to ensure all GPU operations complete
    torch.cuda.synchronize()     
    
    # Force garbage collection of Python objects
    gc.collect()    
    
    # Clear PyTorch CUDA cache
    torch.cuda.empty_cache()

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

def remove_floater(mesh):
    print('Removing floater ...')
    faces = mesh.faces.cpu().numpy()
    print(f"Current faces: {len(faces)}")
    mesh_set = pymeshlab.MeshSet()
    mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=mesh.vertices.cpu().numpy(), face_matrix=faces)
    mesh_set.add_mesh(mesh_pymeshlab, "converted_mesh")
    mesh_set = pymeshlab_remove_floater(mesh_set)
    
    mesh_pymeshlab = mesh_set.current_mesh()    
    
    new_faces = mesh_pymeshlab.face_matrix()
    print(f"After removing floater: {len(new_faces)}")
    
    new_vertices = torch.from_numpy(mesh_pymeshlab.vertex_matrix()).contiguous().float()
    new_faces = torch.from_numpy(new_faces).contiguous().int()   
    
    mesh.vertices = new_vertices
    mesh.faces = new_faces
    
    return mesh
    
def remove_floater2(vertices, faces):
    print('Removing floater ...')
    #faces = faces.cpu().numpy()
    print(f"Current faces: {len(faces)}")
    mesh_set = pymeshlab.MeshSet()
    mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    mesh_set.add_mesh(mesh_pymeshlab, "converted_mesh")
    mesh_set = pymeshlab_remove_floater(mesh_set)
    
    mesh_pymeshlab = mesh_set.current_mesh()    
    
    new_faces = mesh_pymeshlab.face_matrix()
    print(f"After removing floater: {len(new_faces)}")
    
    new_vertices = mesh_pymeshlab.vertex_matrix()
    
    return new_vertices, new_faces

def remove_mesh_infinite_vertices(mesh):
    print('Removing infinite vertices ...')
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    
    trimesh = Trimesh.Trimesh(vertices=vertices,faces=faces)
    print(f"Original vertex count: {len(trimesh.vertices)}")
    
    # Remove anything outside a reasonable bounding box
    limit = 1e10 
    valid_mask = (np.abs(trimesh.vertices) < limit).all(axis=1)
    
    trimesh.update_vertices(valid_mask)
    
    # Removing vertices can leave "degenerate" faces or orphan nodes
    trimesh.update_faces(trimesh.nondegenerate_faces())
    trimesh.remove_unreferenced_vertices()  

    print(f"Cleaned vertex count: {len(trimesh.vertices)}")
    
    new_vertices = torch.from_numpy(trimesh.vertices).float()
    new_faces = torch.from_numpy(trimesh.faces).int()   
    
    mesh.vertices = new_vertices
    mesh.faces = new_faces    
    
    return mesh
    
def pymeshlab_remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=0.005)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh 
    
def _batched_unsigned_distance(bvh, positions, batch_size=100000, return_uvw=False):
    """
    Batch unsigned_distance queries to avoid GPU kernel timeout on large meshes.
    When processing high-resolution textures (e.g., 2048x2048 = ~4M pixels) on complex
    meshes, a single BVH query can cause GPU watchdog timeout. This function splits
    the query into smaller batches.
    Args:
        bvh: The BVH structure from cumesh
        positions: (N, 3) tensor of query positions
        batch_size: Maximum number of queries per batch (default 100K, matching
            the rasterization chunk size used elsewhere in this file)
        return_uvw: Whether to return barycentric coordinates
    Returns:
        Same as bvh.unsigned_distance()
    """
    import torch
    N = positions.shape[0]
    if N <= batch_size:
        return bvh.unsigned_distance(positions, return_uvw=return_uvw)

    distances_list = []
    face_id_list = []
    uvw_list = [] if return_uvw else None

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        d, f, u = bvh.unsigned_distance(positions[i:end], return_uvw=return_uvw)
        distances_list.append(d)
        face_id_list.append(f)
        if return_uvw:
            uvw_list.append(u)

    return (
        torch.cat(distances_list),
        torch.cat(face_id_list),
        torch.cat(uvw_list) if return_uvw else None
    )    

class Trellis2LoadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelname": (["microsoft/TRELLIS.2-4B","visualbruno/TRELLIS.2-4B-FP8","TencentARC/Pixal3D-T"],{"default":"microsoft/TRELLIS.2-4B"}),
                "backend": (["flash_attn","xformers","sdpa","flash_attn_3"],{"default":"flash_attn"}),
                "device": (["cpu","cuda"],{"default":"cuda"}),
                "low_vram": ("BOOLEAN",{"default":True}),
                "keep_models_loaded": ("BOOLEAN", {"default":True}),
                "conv_backend": (["spconv","torchsparse","flex_gemm"],{"default":"flex_gemm"}),
                "sparse_backend": (["xformers","flash_attn"],{"default":"flash_attn"}),
                "use_reconviagen": ("BOOLEAN",{"default":False}),
                #"naf_chunk_size":(["None","144","208","272","336","400","464","528","592","656","720","784","848","912","976","1024"],{"default":"None"}),
            }
        }

    RETURN_TYPES = ("TRELLIS2PIPELINE", )
    RETURN_NAMES = ("pipeline", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, modelname, backend, device, low_vram, keep_models_loaded, conv_backend, sparse_backend, use_reconviagen):    
        import requests
        
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
        #os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
        #os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'        
        os.environ['ATTN_BACKEND'] = backend
        
        config.set_backend(backend)
        sparseconfig.set_attn_backend(sparse_backend)
        sparseconfig.set_conv_backend(conv_backend)
        
        reset_cuda()
        
        torch.backends.cudnn.benchmark = False        
            
        model_path = os.path.join(folder_paths.models_dir, modelname)
        
        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=modelname,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
        
        reconviagen_pipeline_file = os.path.join(folder_paths.models_dir,'microsoft','TRELLIS.2-4B','reconviagen_pipeline.json')
        if not os.path.exists(reconviagen_pipeline_file):
            source_reconviagen_pipeline_file = os.path.join(script_directory,'reconviagen_pipeline.json')
            shutil.copyfile(source_reconviagen_pipeline_file,reconviagen_pipeline_file)
        
        dinov3_model_path = os.path.join(folder_paths.models_dir,"facebook","dinov3-vitl16-pretrain-lvd1689m","model.safetensors")
        if not os.path.exists(dinov3_model_path):
            print('Downloading dinov3 model files ...')            
            from huggingface_hub import hf_hub_download
            
            repo_id = "visualbruno/dinov3-vitl16-pretrain-lvd1689m"
            target_dir = os.path.join(folder_paths.models_dir, "facebook", "dinov3-vitl16-pretrain-lvd1689m")
            
            os.makedirs(target_dir, exist_ok=True)

            # Enable the fast Rust-based transfer backend (multi-connection, much faster)
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

            for fname in ["model.safetensors", "config.json", "preprocessor_config.json"]:
                print(f"downloading {fname} ...")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                )
                print(f"{fname} -> {downloaded_path}")            
        
        trellis_image_large_path = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS-image-large","ckpts","ss_dec_conv3d_16l8_fp16.safetensors")
        if not os.path.exists(trellis_image_large_path):
            print('Trellis-Image-Large ss_dec_conv3d_16l8_fp16 files not found. Trying to download the files from huggingface ...')            
            url = "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.json?download=true"
            filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS-image-large","ckpts","ss_dec_conv3d_16l8_fp16.json")
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
            filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS-image-large","ckpts","ss_dec_conv3d_16l8_fp16.safetensors")

            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print("Download ss_dec_conv3d_16l8_fp16.safetensors complete!")
            else:
                raise Exception("Cannot download Trellis-Image-Large file ss_dec_conv3d_16l8_fp16.safetensors")
        
        if use_reconviagen and modelname == 'TencentARC/Pixal3D-T':
            raise Exception('Model TencentARC/Pixal3D-T is not compatible with ReconViaGen')
        
        if use_reconviagen:
            reconviagen_file = os.path.join(folder_paths.models_dir,'microsoft','TRELLIS.2-4B','ckpts','ss_vggt_cond.safetensors')
            if not os.path.exists(reconviagen_file):
                print('ReconViaGen file ss_vggt_cond.safetensors not found. Trying to download the files from huggingface ...')            
                url = "https://huggingface.co/Stable-X/trellis-vggt-v0-2/resolve/main/ckpts/ss_vggt_cond.safetensors?download=true"
                filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS.2-4B","ckpts","ss_vggt_cond.safetensors")
                path = Path(filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print("Download ss_vggt_cond.safetensors complete!")
                else:
                    raise Exception("Cannot download ReconViaGen file ss_vggt_cond.safetensors")
            
            reconviagen_file = os.path.join(folder_paths.models_dir,'microsoft','TRELLIS.2-4B','ckpts','ss_vggt_cond.json')
            if not os.path.exists(reconviagen_file):
                print('ReconViaGen file ss_vggt_cond.json not found. Trying to download the files from huggingface ...')            
                url = "https://huggingface.co/Stable-X/trellis-vggt-v0-2/resolve/main/ckpts/ss_vggt_cond.json?download=true"
                filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS.2-4B","ckpts","ss_vggt_cond.json")
                path = Path(filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print("Download ss_vggt_cond.json complete!")
                else:
                    raise Exception("Cannot download ReconViaGen file ss_vggt_cond.json") 

            reconviagen_file = os.path.join(folder_paths.models_dir,'microsoft','TRELLIS.2-4B','ckpts','ss_flow_img_dit_L_16l8_fp16.safetensors')
            if not os.path.exists(reconviagen_file):
                print('ReconViaGen file ss_flow_img_dit_L_16l8_fp16.safetensors not found. Trying to download the files from huggingface ...')            
                url = "https://huggingface.co/Stable-X/trellis-vggt-v0-2/resolve/main/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors?download=true"
                filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS.2-4B","ckpts","ss_flow_img_dit_L_16l8_fp16.safetensors")
                path = Path(filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print("Download ss_flow_img_dit_L_16l8_fp16.safetensors complete!")
                else:
                    raise Exception("Cannot download ReconViaGen file ss_flow_img_dit_L_16l8_fp16.safetensors")      

            reconviagen_file = os.path.join(folder_paths.models_dir,'microsoft','TRELLIS.2-4B','ckpts','ss_flow_img_dit_L_16l8_fp16.json')
            if not os.path.exists(reconviagen_file):
                print('ReconViaGen file ss_flow_img_dit_L_16l8_fp16.json not found. Trying to download the files from huggingface ...')            
                url = "https://huggingface.co/Stable-X/trellis-vggt-v0-2/resolve/main/ckpts/ss_flow_img_dit_L_16l8_fp16.json?download=true"
                filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS.2-4B","ckpts","ss_flow_img_dit_L_16l8_fp16.json")
                path = Path(filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print("Download ss_flow_img_dit_L_16l8_fp16.json complete!")
                else:
                    raise Exception("Cannot download ReconViaGen file ss_flow_img_dit_L_16l8_fp16.json")                       
        
        if modelname == "visualbruno/TRELLIS.2-4B-FP8":
            use_fp8 = True
            if use_reconviagen:
                raise Exception("ReconViaGen cannot be used with TRELLIS.2-4B-FP8. Select microsoft/TRELLIS.2-4B")
        else:
            use_fp8 = False
        
        isPixal3D = False
        if modelname == "TencentARC/Pixal3D-T":
            isPixal3D = True
        
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path, keep_models_loaded = keep_models_loaded, use_fp8=use_fp8, use_reconviagen=use_reconviagen, isPixal3D = isPixal3D)
        # Pixal3D NAFF model uses NATTEN for feature upsampling.
        # Windows prebuilt NATTEN wheels (torch 2.7.0, cu128, cp312):
        #   https://hf-mirror.com/lldacing/NATTEN-windows
        # For issues/questions about Pixal3D Windows setup:
        #   https://space.bilibili.com/37411464
        pipeline.low_vram = low_vram
        
        # if naf_chunk_size == "None":
            # pipeline.naf_chunk_size = None
        # else:
            # pipeline.naf_chunk_size = int(naf_chunk_size)
        
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "pipeline_type": (["512","1024","1024_cascade","1536_cascade"],{"default":"1024_cascade"}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "max_num_tokens": ("INT",{"default":49152,"min":0,"max":999999}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "generate_texture_slat": ("BOOLEAN", {"default":True}),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", "BVH", )
    RETURN_NAMES = ("mesh", "bvh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, seed, pipeline_type, sparse_structure_steps, shape_steps, texture_steps, max_num_tokens, max_views, sparse_structure_resolution, generate_texture_slat, use_tiled_decoder, sampler, fill_holes, hole_iterations, hole_fill_algorithm, keep_only_shell):
        reset_cuda()
        
        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images
        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps}
        shape_slat_sampler_params = {"steps":shape_steps}
        tex_slat_sampler_params = {"steps":texture_steps}
        
        if generate_texture_slat:
            num_steps = 5
        else:
            num_steps = 4

        pbar = ProgressBar(num_steps)        
        
        mesh = pipeline.run(image=image_in, 
                            seed=seed, 
                            pipeline_type=pipeline_type, 
                            sparse_structure_sampler_params = sparse_structure_sampler_params, 
                            shape_slat_sampler_params = shape_slat_sampler_params, 
                            tex_slat_sampler_params = tex_slat_sampler_params, 
                            max_num_tokens = max_num_tokens, 
                            sparse_structure_resolution = sparse_structure_resolution, 
                            max_views = max_views, 
                            generate_texture_slat = generate_texture_slat, 
                            use_tiled=use_tiled_decoder, 
                            pbar=pbar, 
                            sampler=sampler,
                            fill_holes=fill_holes,
                            hole_iterations=hole_iterations,
                            hole_fill_algorithm=hole_fill_algorithm,
                            keep_only_shell=keep_only_shell)[0]
        
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()        
       
        # Build BVH for the current mesh to guide remeshing
        if generate_texture_slat:
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())           
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()
        else:
            print("Not building BVH : only used for texturing")
            bvh = None        
        
        return (mesh, bvh,)    

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
                "target_face_num": ("INT",{"default":1000000,"min":1,"max":30000000}),
                "method": (["Cumesh","Meshlib"],{"default":"Cumesh"}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, target_face_num, method):        
        mesh_copy = copy.deepcopy(mesh)
        if method=="Cumesh":
            # internal testing future release
            # options = {
                # 'method': 'legacy'
            # }             
            mesh_copy.simplify_with_cumesh(target = target_face_num)
        elif method=="Meshlib":
            mesh_copy.simplify_with_meshlib(target = target_face_num)
        else:
            raise Exception("Unknown simplification method")             
        
        return (mesh_copy,)     

class Trellis2SimplifyTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_face_num": ("INT",{"default":1000000,"min":1,"max":30000000}),
                "method": (["Cumesh","Meshlib"],{"default":"Cumesh"}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, target_face_num, method):        
        mesh_copy = copy.deepcopy(trimesh)
        if method=="Cumesh":
            # internal testing future release
            # options = {
                # 'options': 'legacy'
            # }            
            cumesh = CuMesh.CuMesh()
            cumesh.init(torch.from_numpy(mesh_copy.vertices).float().cuda(), torch.from_numpy(mesh_copy.faces).int().cuda())
            cumesh.simplify(target_face_num, verbose=True)
            new_vertices, new_faces = cumesh.read()
            mesh_copy.vertices = new_vertices.cpu().numpy()
            mesh_copy.faces = new_faces.cpu().numpy()
            
            del cumesh
        elif method=="Meshlib":
            new_vertices, new_faces = simplify_with_meshlib(mesh_copy.vertices, mesh_copy.faces, target = target_face_num)
            mesh_copy.vertices = new_vertices
            mesh_copy.faces = new_faces
        else:
            raise Exception("Unknown simplification method")             
        
        return (mesh_copy,)   

class Trellis2ProgressiveSimplify:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_edge_length": ("FLOAT",{"default":0.00,"min":0.00,"max":99999.99,"step":0.01}),
                "max_triangle_aspect_ratio": ("FLOAT",{"default":20.00,"min":0.01,"max":99999.99,"step":0.01}),
                "strategy": (["Minimal Error First","Shortest Edge First"],{"default":"Minimal Error First"}),
                "stabilizer": ("FLOAT",{"default":0.000001,"min":0.0,"max":0.999999,"step":0.000001}),
                "touch_near_boundary_edges": ("BOOLEAN",{"default":True}),
                "optimize_vertex_positions": ("BOOLEAN",{"default":True}),
                "angle_based_weights": ("BOOLEAN",{"default":False}),                
            },
            "optional": {
                "trimesh": ("TRIMESH",),
                "mesh": ("MESHWITHVOXEL",),
            }
        }

    RETURN_TYPES = ("TRIMESH", "MESHWITHVOXEL",)
    RETURN_NAMES = ("trimesh", "mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, max_edge_length, max_triangle_aspect_ratio, strategy, stabilizer, touch_near_boundary_edges, optimize_vertex_positions, angle_based_weights, trimesh = None, mesh = None):        
        if trimesh is not None:
            trimesh = copy.deepcopy(trimesh)
            
            vertices = trimesh.vertices
            faces = trimesh.faces
            
            vertices, faces = self.simplify(vertices, faces, max_edge_length, max_triangle_aspect_ratio, strategy, stabilizer, touch_near_boundary_edges, optimize_vertex_positions, angle_based_weights)
            trimesh.vertices = vertices
            trimesh.faces = faces
            
        if mesh is not None:
            mesh = copy.deepcopy(mesh)
            
            vertices = mesh.vertices.cpu().numpy()
            faces = mesh.faces.cpu().numpy()
            
            vertices, faces = self.simplify(vertices, faces, max_edge_length, max_triangle_aspect_ratio, strategy, stabilizer, touch_near_boundary_edges, optimize_vertex_positions, angle_based_weights)
            mesh.vertices = torch.from_numpy(vertices).float()
            mesh.faces = torch.from_numpy(faces).int()
        
        return (trimesh, mesh) 

    def simplify(self, vertices, faces, max_edge_length, max_triangle_aspect_ratio, strategy, stabilizer, touch_near_boundary_edges, optimize_vertex_positions, angle_based_weights):
        current_faces_num = len(faces)
        print(f'Current Faces Number: {current_faces_num}')

        settings = mrmeshpy.DecimateSettings()
        
        if strategy == "Minimal Error First":
            settings.strategy = mrmeshpy.DecimateStrategy.MinimizeError
        else:
            settings.strategy = mrmeshpy.DecimateStrategy.ShortestEdgeFirst
            
        settings.maxTriangleAspectRatio = max_triangle_aspect_ratio
        settings.stabilizer = stabilizer
        settings.touchNearBdEdges = touch_near_boundary_edges
        settings.optimizeVertexPos = optimize_vertex_positions
        settings.angleWeightedDistToPlane = angle_based_weights        
        settings.packMesh = True
        
        print('Generating Meshlib Mesh ...')
        mesh = mrmeshnumpy.meshFromFacesVerts(faces, vertices)
        
        if max_edge_length == 0.0:
            max_edge_length = 2.0
            # for edge_id in mesh.topology.allValidEdges():
                    # edge_len = mesh.computeEdgeLen(edge_id)
                    # if edge_len > max_edge_length:
                        # max_edge_length = edge_len
            # print(f"Calculated Max Edge Length: {max_edge_length}")
            
        settings.maxEdgeLen = max_edge_length   
        settings.maxError = max_edge_length / 1000
        
        print('Packing Optimally ...')
        mesh.packOptimally()
        print('Decimating ...')
        mrmeshpy.decimateMesh(mesh, settings)
        
        new_vertices = mrmeshnumpy.getNumpyVerts(mesh)
        new_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)               
        
        print(f"Reduced faces, resulting in {len(new_vertices)} vertices and {len(new_faces)} faces")
            
        del mesh
        gc.collect()
        
        return new_vertices, new_faces        
        
class Trellis2MeshWithVoxelToTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "reorient_vertices":(["None","90 degrees","-90 degrees"],{"default":"90 degrees"}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, reorient_vertices):       
        mesh_copy = copy.deepcopy(mesh)
        
        vertices_np = mesh_copy.vertices.cpu().numpy()
        
        if reorient_vertices == '90 degrees':
            vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        elif reorient_vertices == '-90 degrees':
            vertices_np[:, 1], vertices_np[:, 2] = -vertices_np[:, 2], vertices_np[:, 1]
        
        trimesh = Trimesh.Trimesh(
            vertices=vertices_np,
            faces=mesh_copy.faces.cpu().numpy(),
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
            }
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("glb_path","relative_path",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, filename_prefix, file_format):        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())                      
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(exist_ok=True)

        if file_format=='obj':
            materialName = f"{filename}_{counter:05}_.mtl"
            if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material') and trimesh.visual.material is not None:
                trimesh.visual.material.name = f"{filename}_{counter:05}"

            trimesh.export(output_glb_path, file_type=file_format, mtl_name=materialName)
        else:
            trimesh.export(output_glb_path, file_type=file_format)
            
        relative_path = Path(subfolder) / f'{filename}_{counter:05}_.{file_format}'
        
        return (str(output_glb_path), str(relative_path), )        
        
class Trellis2PostProcessMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "remove_duplicate_faces": ("BOOLEAN",{"default":False}),
                "repair_non_manifold_edges": ("BOOLEAN", {"default":False}),
                "remove_non_manifold_faces": ("BOOLEAN", {"default":False}),
                "remove_small_connected_components": ("BOOLEAN", {"default":False}),
                "remove_small_connected_components_size": ("FLOAT", {"default":0.00001,"min":0.00001,"max":9.99999,"step":0.00001}),
                "unify_faces_orientation": ("BOOLEAN", {"default":False}),
                "remove_floaters": ("BOOLEAN",{"default":False}),
                "remove_infinite_vertices": ("BOOLEAN",{"default":False}),
                "merge_vertices": ("BOOLEAN",{"default":False}),
                "merge_distance": ("FLOAT",{"default":0.0010,"min":0.0001,"max":999.9999,"step":0.0001}),
                "remove_nan_vertices": ("BOOLEAN",{"default":False}),                
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, 
        mesh, 
        remove_duplicate_faces, 
        repair_non_manifold_edges, 
        remove_non_manifold_faces, 
        remove_small_connected_components, 
        remove_small_connected_components_size,
        unify_faces_orientation,
        remove_floaters,
        remove_infinite_vertices,
        merge_vertices,
        merge_distance,
        remove_nan_vertices):
            
        mesh_copy = copy.deepcopy(mesh)

        if remove_floaters:
            mesh_copy = remove_floater(mesh_copy)
        if remove_infinite_vertices:
            mesh_copy = remove_mesh_infinite_vertices(mesh_copy)                    

        vertices = mesh_copy.vertices
        faces = mesh_copy.faces

        # Move data to GPU
        vertices = vertices.cuda()
        faces = faces.cuda()
        
        # Initialize CUDA mesh handler
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        print(f"Current vertices: {cumesh.num_vertices}, faces: {cumesh.num_faces}")
            
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
        
        if unify_faces_orientation:
            print('Unifying faces orientation ...')
            cumesh.unify_face_orientations()            
        
        if merge_vertices or remove_nan_vertices:
            import open3d            
            open3d_mesh = open3d.geometry.TriangleMesh()
            open3d_mesh.vertices = open3d.utility.Vector3dVector(vertices.cpu().numpy())
            open3d_mesh.triangles = open3d.utility.Vector3iVector(faces.cpu().numpy().astype(np.int32))

            # NaN check
            print('Removing NaN vertices ...')
            verts = np.asarray(open3d_mesh.vertices)
            if np.any(np.isnan(verts)) or np.any(np.isinf(verts)):
                print('NaN found. Cleaning them ...')
                verts = np.nan_to_num(verts, nan=0.0, posinf=0.0, neginf=0.0)
                open3d_mesh.vertices = open3d.utility.Vector3dVector(verts)
                open3d_mesh = open3d_mesh.remove_duplicated_vertices()
                open3d_mesh = open3d_mesh.remove_duplicated_triangles()
                open3d_mesh = open3d_mesh.remove_degenerate_triangles()
                open3d_mesh = open3d_mesh.remove_unreferenced_vertices()

            #bbox = open3d_mesh.get_axis_aligned_bounding_box()
            #max_extent = np.max(bbox.get_extent())
            #safe_merge_distance = max_extent * 0.0005  # More conservative
            #print(f"Auto-calculated merge distance: {safe_merge_distance:.6f}")
            
            if merge_vertices:
                # Merge and cleanup
                open3d_mesh = open3d_mesh.merge_close_vertices(merge_distance)
                open3d_mesh = open3d_mesh.remove_duplicated_vertices()
                open3d_mesh = open3d_mesh.remove_duplicated_triangles()
                open3d_mesh = open3d_mesh.remove_degenerate_triangles()
                open3d_mesh = open3d_mesh.remove_unreferenced_vertices()

                # Proper normal computation sequence
                open3d_mesh.compute_triangle_normals()
                open3d_mesh.compute_vertex_normals()
                open3d_mesh.normalize_normals()
                open3d_mesh.orient_triangles()  # Orient based on computed normals
                open3d_mesh.compute_vertex_normals()  # Recompute after orientation

                # Gentler smoothing
                open3d_mesh = open3d_mesh.filter_smooth_taubin(number_of_iterations=3)
                open3d_mesh.compute_vertex_normals()  # Final recompute            
            
            cumesh.init(torch.from_numpy(np.asarray(open3d_mesh.vertices)).cuda().float(), torch.from_numpy(np.asarray(open3d_mesh.triangles)).cuda().int())
            del open3d_mesh
        
        print(f"After initial cleanup: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")                                                   
        
        new_vertices, new_faces = cumesh.read()
        
        mesh_copy.vertices = new_vertices.to(mesh_copy.device)
        mesh_copy.faces = new_faces.to(mesh_copy.device) 
        
        del cumesh
        gc.collect()
                
        return (mesh_copy,)
       
class Trellis2UnWrapAndRasterizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60.0,"min":0.0,"max":359.9}),
                "mesh_cluster_refine_iterations": ("INT",{"default":0}),
                "mesh_cluster_global_iterations": ("INT",{"default":1}),
                "mesh_cluster_smooth_strength": ("INT",{"default":1}),                
                "texture_size": ("INT",{"default":4096, "min":512, "max":16384}),
                "texture_alpha_mode": (["OPAQUE","MASK","BLEND"],{"default":"OPAQUE"}),
                "double_side_material": ("BOOLEAN",{"default":False}),
                "bake_on_vertices": ("BOOLEAN",{"default":False}),
                "use_custom_normals": ("BOOLEAN",{"default":False}),
                "bvh": ("BVH",),
                "inpainting": (["telea","ns"],{"default":"telea"}),
                "reorient_vertices":(["None","90 degrees","-90 degrees"],{"default":"90 degrees"}),
            }
        }

    RETURN_TYPES = ("TRIMESH","IMAGE","IMAGE",)
    RETURN_NAMES = ("trimesh","base_color_texture", "metallic_roughness_texture",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, mesh_cluster_threshold_cone_half_angle_rad, mesh_cluster_refine_iterations, mesh_cluster_global_iterations, mesh_cluster_smooth_strength, texture_size, texture_alpha_mode, double_side_material, bake_on_vertices,use_custom_normals,bvh,inpainting,reorient_vertices):
        mesh_copy = copy.deepcopy(mesh)
        
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        
        vertices = mesh_copy.vertices
        faces = mesh_copy.faces
        attr_volume = mesh_copy.attrs
        coords = mesh_copy.coords
        attr_layout = mesh_copy.layout
        voxel_size = mesh_copy.voxel_size  
        
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
        # if bvh == None:
        # print(f"Building BVH for current mesh...")
        # bvh = CuMesh.cuBVH(vertices, faces) 
        # bvh.vertices = vertices
        # bvh.faces = faces
        
        # --- Branch: Bake On Vertices (skip UV unwrapping and texture creation) ---
        if bake_on_vertices:
            print('Baking colors on vertices...')
            out_vertices, out_faces = cumesh.read()
            out_vertices = out_vertices.cuda()
            out_faces = out_faces.cuda()
            cumesh.compute_vertex_normals()
            out_normals = cumesh.read_vertex_normals()
            
            # Sample attributes directly at vertex positions from the voxel grid
            # No BVH mapping needed - the voxel grid contains all the color information
            vertex_attrs = grid_sample_3d(
                attr_volume,
                torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
                shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
                grid=((out_vertices - aabb[0]) / voxel_size).reshape(1, -1, 3),
                mode='trilinear',
            )
            
            # Extract base color and alpha per vertex (vertex_attrs shape: N_vertices x C)
            base_color_idx = attr_layout['base_color']
            alpha_idx = attr_layout['alpha']
            
            # Get RGB values and squeeze any extra dimensions to get (N, 3)
            vertex_colors_rgb = vertex_attrs[..., base_color_idx].cpu().numpy()
            vertex_colors_rgb = np.squeeze(vertex_colors_rgb)  # Remove batch dims if any
            if vertex_colors_rgb.ndim == 1:
                vertex_colors_rgb = vertex_colors_rgb[None, :]  # Ensure at least 2D
            vertex_colors_rgb = np.clip(vertex_colors_rgb * 255, 0, 255).astype(np.uint8)
            
            # Handle alpha based on texture_alpha_mode
            if texture_alpha_mode == "OPAQUE":
                # For OPAQUE mode, use full alpha (255)
                vertex_alpha = np.full((vertex_colors_rgb.shape[0], 1), 255, dtype=np.uint8)
            else:
                vertex_alpha = vertex_attrs[..., alpha_idx].cpu().numpy()
                vertex_alpha = np.squeeze(vertex_alpha)  # Remove batch dims if any
                vertex_alpha = np.clip(vertex_alpha * 255, 0, 255).astype(np.uint8)
                # Ensure alpha is 2D with shape (N, 1)
                if vertex_alpha.ndim == 1:
                    vertex_alpha = vertex_alpha[:, None]
            
            # Combine into RGBA
            vertex_colors_rgba = np.concatenate([vertex_colors_rgb, vertex_alpha], axis=-1)
            
            print("Finalizing mesh with vertex colors...")
            
            vertices_np = out_vertices.cpu().numpy()
            faces_np = out_faces.cpu().numpy()
            normals_np = out_normals.cpu().numpy()
            
            # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
            if reorient_vertices == '90 degrees':
                vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
                normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
            elif reorient_vertices == '-90 degrees':
                vertices_np[:, 1], vertices_np[:, 2] = -vertices_np[:, 2], vertices_np[:, 1]
                normals_np[:, 1], normals_np[:, 2] = -normals_np[:, 2], normals_np[:, 1]
            
            # Create mesh with vertex colors using ColorVisuals
            if use_custom_normals:
                textured_mesh = Trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_normals=normals_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )
            else:
                textured_mesh = Trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )                
            
            del cumesh
            gc.collect()
            
            # Return empty placeholder textures for vertex color mode
            placeholder_texture = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
            return (textured_mesh, placeholder_texture, placeholder_texture,)        
        
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
        orig_tri_verts = bvh.vertices[bvh.faces[face_id.long()]] # (N_new, 3, 3)
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)        
        
        torch.cuda.synchronize()
        
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
        
        if inpainting == 'telea':
            inpainting = cv2.INPAINT_TELEA
        else:
            inpainting = cv2.INPAINT_NS
        
        # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
        mask_inv = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask_inv, 3, inpainting)
        metallic = inpaint_channel(metallic, mask_inv, 1, inpainting)
        roughness = inpaint_channel(roughness, mask_inv, 1, inpainting)
        alpha = inpaint_channel(alpha, mask_inv, 1, inpainting)
        
        # Create PBR material
        # Standard PBR packs Metallic and Roughness into Blue and Green channels
        baseColorTexture_np = Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
        metallicRoughnessTexture_np = Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1))
        
        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=baseColorTexture_np,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=metallicRoughnessTexture_np,
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
        if reorient_vertices == '90 degrees':
            vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
            normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        elif reorient_vertices == '-90 degrees':
            vertices_np[:, 1], vertices_np[:, 2] = -vertices_np[:, 2], vertices_np[:, 1]
            normals_np[:, 1], normals_np[:, 2] = -normals_np[:, 2], normals_np[:, 1]        
            
        uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
        
        if use_custom_normals:
            textured_mesh = Trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                vertex_normals=normals_np,
                process=False,
                visual=Trimesh.visual.TextureVisuals(uv=uvs_np,material=material)
            )
        else:
            textured_mesh = Trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                process=False,
                visual=Trimesh.visual.TextureVisuals(uv=uvs_np,material=material)
            )            

        del cumesh
        gc.collect()    

        baseColorTexture = pil2tensor(baseColorTexture_np)
        metallicRoughnessTexture = pil2tensor(metallicRoughnessTexture_np)
                
        return (textured_mesh, baseColorTexture, metallicRoughnessTexture, )
        
class Trellis2MeshWithVoxelAdvancedGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0x7fffffff}),
                "pipeline_type": (["512","1024","1024_cascade","1536_cascade"],{"default":"1024_cascade"}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":3.00,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.20,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":3.00,"min":0.00,"max":9.99,"step":0.01}),                
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "generate_texture_slat": ("BOOLEAN", {"default":True}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),                
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL","BVH", )
    RETURN_NAMES = ("mesh", "bvh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, seed, pipeline_type, sparse_structure_steps, 
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
        max_num_tokens,
        max_views,
        sparse_structure_resolution,
        generate_texture_slat,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        use_tiled_decoder,
        sampler,
        fill_holes,
        hole_iterations,
        verbose,
        dino_lock,
        dino_substeps,
        hole_fill_algorithm,
        dino_foundation_cap,
        keep_only_shell
        ):
        reset_cuda()
        
        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images
        
        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]
        shape_guidance_interval = [shape_guidance_interval_start,shape_guidance_interval_end]
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]
        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}       
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
            
        if generate_texture_slat:
            num_steps = 5
        else:
            num_steps = 4        
        
        if pipeline.isPixal3D:
            pipeline.load_moge_model()
            
            if isinstance(images, (list, tuple)):
                image = images[0]
            else:
                image = images
                
            camera_params = pipeline.get_moge_camera_config(image)

            if not pipeline.keep_models_loaded: 
                pipeline.unload_moge_model()
            
            mesh = pipeline.run_pixal3d(image=image_in, 
                                seed=seed, 
                                pipeline_type=pipeline_type, 
                                sparse_structure_sampler_params = sparse_structure_sampler_params, 
                                shape_slat_sampler_params = shape_slat_sampler_params, 
                                tex_slat_sampler_params = tex_slat_sampler_params, 
                                max_num_tokens = max_num_tokens,
                                generate_texture_slat=generate_texture_slat,
                                camera_params=camera_params)[0]             
        else:
            pbar = ProgressBar(num_steps)
            mesh = pipeline.run(image=image_in, 
                                seed=seed, 
                                pipeline_type=pipeline_type, 
                                sparse_structure_sampler_params = sparse_structure_sampler_params, 
                                shape_slat_sampler_params = shape_slat_sampler_params, 
                                tex_slat_sampler_params = tex_slat_sampler_params, 
                                max_num_tokens = max_num_tokens, 
                                sparse_structure_resolution = sparse_structure_resolution, 
                                max_views = max_views, 
                                generate_texture_slat=generate_texture_slat, 
                                use_tiled=use_tiled_decoder, 
                                pbar=pbar, 
                                sampler=sampler,
                                fill_holes = fill_holes,
                                hole_iterations = hole_iterations,
                                verbose = verbose,
                                dino_lock = dino_lock,
                                dino_substeps = dino_substeps,
                                hole_fill_algorithm=hole_fill_algorithm,
                                dino_foundation_cap=dino_foundation_cap,
                                keep_only_shell=keep_only_shell)[0]         
        
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()                
        
        if generate_texture_slat:
            # Build BVH for the current mesh to guide remeshing
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())           
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()
        else:
            print("Not building BVH : only used for texturing")
            bvh = None
        
        return (mesh,bvh,)         

class Trellis2MeshWithVoxelMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "front_image": ("IMAGE",),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0x7fffffff}),
                "pipeline_type": (["512","1024","1024_cascade","1536_cascade"],{"default":"1024_cascade"}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":3.00,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.20,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":3.00,"min":0.00,"max":9.99,"step":0.01}),                 
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "generate_texture_slat": ("BOOLEAN", {"default":True}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),                
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
            "optional": {
                "back_image": ("IMAGE",),
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL","BVH", )
    RETURN_NAMES = ("mesh", "bvh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, front_image, seed, pipeline_type, sparse_structure_steps, 
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
        max_num_tokens,
        sparse_structure_resolution,
        generate_texture_slat,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        use_tiled_decoder,
        front_axis,
        blend_temperature,
        sampler,
        fill_holes,
        hole_iterations,
        verbose,
        dino_lock,
        dino_substeps,
        hole_fill_algorithm,
        dino_foundation_cap,
        keep_only_shell,
        back_image=None,
        left_image=None,
        right_image=None):

        reset_cuda()
        
        # Convert front image tensor to PIL
        front_pil = tensor2pil(front_image)
        
        # Convert optional view image tensors to PIL
        back_pil = tensor2pil(back_image) if back_image is not None else None
        left_pil = tensor2pil(left_image) if left_image is not None else None
        right_pil = tensor2pil(right_image) if right_image is not None else None        
        
        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]
        shape_guidance_interval = [shape_guidance_interval_start,shape_guidance_interval_end]
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]
        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}       
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
            
        if generate_texture_slat:
            num_steps = 5
        else:
            num_steps = 4

        pbar = ProgressBar(num_steps)
        
        mesh = pipeline.run_multiview(
            front=front_pil,
            back=back_pil,
            left=left_pil,
            right=right_pil,
            seed=seed,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            shape_slat_sampler_params=shape_slat_sampler_params,
            tex_slat_sampler_params=tex_slat_sampler_params,
            max_num_tokens=max_num_tokens,
            sparse_structure_resolution=sparse_structure_resolution,
            generate_texture_slat=generate_texture_slat,
            use_tiled=use_tiled_decoder,
            pbar=pbar,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            sampler=sampler,
            fill_holes=fill_holes,
            hole_iterations=hole_iterations,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            hole_fill_algorithm=hole_fill_algorithm,
            dino_foundation_cap=dino_foundation_cap,
            keep_only_shell=keep_only_shell
        )[0]         
        
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()                
        
        if generate_texture_slat:
            # Build BVH for the current mesh to guide remeshing
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())           
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()
        else:
            print("Not building BVH : only used for texturing")
            bvh = None
        
        return (mesh,bvh,)

class Trellis2PostProcessAndUnWrapAndRasterizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60.0,"min":0.0,"max":359.9}),
                "mesh_cluster_refine_iterations": ("INT",{"default":0}),
                "mesh_cluster_global_iterations": ("INT",{"default":1}),
                "mesh_cluster_smooth_strength": ("INT",{"default":1}),                
                "texture_size": ("INT",{"default":4096, "min":512, "max":16384}),
                "remesh": ("BOOLEAN",{"default":True}),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "remesh_project": ("FLOAT",{"default":0.0}),
                "target_face_num": ("INT",{"default":2000000,"min":1,"max":16000000}),
                "simplify_method": (["Cumesh","Meshlib"],{"default":"Cumesh"}),
                "fill_holes": ("BOOLEAN", {"default":True}),
                "texture_alpha_mode": (["OPAQUE","MASK","BLEND"],{"default":"OPAQUE"}),
                "dual_contouring_resolution": (["Auto","128","256","512","1024","2048"],{"default":"1024"}),
                "double_side_material": ("BOOLEAN",{"default":False}),
                "remove_floaters": ("BOOLEAN",{"default":True}),
                "bake_on_vertices": ("BOOLEAN",{"default":False}),
                "use_custom_normals":("BOOLEAN",{"default":False}),
                "bvh": ("BVH",),
                "remove_inner_faces": ("BOOLEAN",{"default":True}),
                "inpainting": (["telea","ns"],{"default":"telea"}),
                "reorient_vertices":(["None","90 degrees","-90 degrees"],{"default":"90 degrees"}),
            }
        }

    RETURN_TYPES = ("TRIMESH","IMAGE","IMAGE",)
    RETURN_NAMES = ("trimesh","base_color_texture","metallic_roughness_texture",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, mesh_cluster_threshold_cone_half_angle_rad, mesh_cluster_refine_iterations, mesh_cluster_global_iterations, mesh_cluster_smooth_strength, texture_size, remesh, remesh_band, remesh_project, target_face_num, simplify_method, fill_holes, texture_alpha_mode, dual_contouring_resolution, double_side_material, remove_floaters, bake_on_vertices,use_custom_normals,bvh,remove_inner_faces,inpainting,reorient_vertices):
        pbar = ProgressBar(5 if not bake_on_vertices else 4)
        mesh_copy = copy.deepcopy(mesh)
        
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        
        attr_volume = mesh_copy.attrs
        coords = mesh_copy.coords
        attr_layout = mesh_copy.layout
        voxel_size = mesh_copy.voxel_size  
        
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
            
        vertices = mesh_copy.vertices
        faces = mesh_copy.faces
        
        vertices = vertices.cuda()
        faces = faces.cuda()                
        
        # Initialize CUDA mesh handler
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        print(f"Current vertices: {cumesh.num_vertices}, faces: {cumesh.num_faces}")        
            
        pbar.update(1)
            
        print("Cleaning mesh...")        
        # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
        if not remesh:            
            if simplify_method == 'Cumesh':
                cumesh.simplify(target_face_num * 3, verbose=True)
            elif simplify_method == 'Meshlib':
                 # GPU -> CPU -> Meshlib -> CPU -> GPU
                v, f = cumesh.read()
                new_vertices, new_faces = simplify_with_meshlib(v.cpu().numpy(), f.cpu().numpy(), target_face_num)
                cumesh.init(torch.from_numpy(new_vertices).float().cuda(), torch.from_numpy(new_faces).int().cuda())        
            
            cumesh.remove_duplicate_faces()
            cumesh.repair_non_manifold_edges()
            cumesh.remove_small_connected_components(1e-5)
            
            if simplify_method == 'Cumesh':
                cumesh.simplify(target_face_num, verbose=True)
            elif simplify_method == 'Meshlib':
                 # GPU -> CPU -> Meshlib -> CPU -> GPU
                v, f = cumesh.read()
                new_vertices, new_faces = simplify_with_meshlib(v.cpu().numpy(), f.cpu().numpy(), target_face_num)
                cumesh.init(torch.from_numpy(new_vertices).float().cuda(), torch.from_numpy(new_faces).int().cuda())
            
            cumesh.remove_duplicate_faces()
            cumesh.repair_non_manifold_edges()
            cumesh.remove_small_connected_components(1e-5)         
            
            print(f"After initial cleanup: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")                            
                
            # Step 2: Unify face orientations
            print('Unifying faces orientation ...')
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
            
            scale = (resolution + 3 * remesh_band) / resolution * scale
            print(f"Calculated scale: {scale}")
            
            print('Performing Dual Contouring ...')
            # Perform Dual Contouring remeshing (rebuilds topology)
            cumesh.init(*CuMesh.remeshing.remesh_narrow_band_dc_quad(
                vertices, faces,
                center = center,
                scale = scale,
                resolution = resolution,
                band = remesh_band,
                project_back = remesh_project, # Snaps vertices back to original surface
                verbose = True,
                remove_inner_faces = remove_inner_faces,
                bvh = bvh,
            ))
            
            new_vertices, new_faces = cumesh.read()
            
            if remove_floaters:
                new_vertices, new_faces = remove_floater2(new_vertices.cpu().numpy(),new_faces.cpu().numpy())
                new_vertices = torch.from_numpy(new_vertices).contiguous().float().cuda()
                new_faces = torch.from_numpy(new_faces).contiguous().int().cuda()
                cumesh.init(new_vertices, new_faces)                    
            
            print(f"After remeshing: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")

            if simplify_method == 'Cumesh':
                cumesh.simplify(target_face_num, verbose=True)
            elif simplify_method == 'Meshlib':
                 # GPU -> CPU -> Meshlib -> CPU -> GPU
                v, f = cumesh.read()
                new_vertices, new_faces = simplify_with_meshlib(v.cpu().numpy(), f.cpu().numpy(), target_face_num)
                cumesh.init(torch.from_numpy(new_vertices).float().cuda(), torch.from_numpy(new_faces).int().cuda())

            print(f"After simplifying: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")            
            pbar.update(1)
            
        if fill_holes:
            new_vertices, new_faces = cumesh.read()
            meshlib_mesh = mrmeshnumpy.meshFromFacesVerts(new_faces.detach().clone().cpu().numpy(), new_vertices.detach().clone().cpu().numpy())
            hole_edges = meshlib_mesh.topology.findHoleRepresentiveEdges()
            holes_filled = 0
            
            nb_holes = len(hole_edges)
            print(f"{nb_holes} holes found")

            if nb_holes > 0:
                progress_bar = tqdm(total=nb_holes, desc="Filling holes")
                
                last_reported_percent = -1  # Initialize at -1 to ensure 0% triggers an update
                
                for i, e in enumerate(hole_edges):
                    params = mrmeshpy.FillHoleParams()
                    params.metric = mrmeshpy.getUniversalMetric(meshlib_mesh)
                    mrmeshpy.fillHole(meshlib_mesh, e, params)
                    
                    # Calculate current progress
                    current_step = i + 1
                    current_percent = int((current_step / nb_holes) * 100)
                    
                    # Only update the UI if the percentage has moved up
                    if current_percent > last_reported_percent:
                        # Calculate how many holes have been filled since the last UI update
                        # This handles cases where 1% might represent multiple holes
                        if last_reported_percent == -1:
                            # First update
                            progress_bar.update(current_step)
                        else:
                            # Update by the difference since the last check
                            last_step = int((last_reported_percent * nb_holes) / 100)
                            diff = current_step - last_step
                            progress_bar.update(diff)
                        
                        last_reported_percent = current_percent
                            
                progress_bar.close()                 
            
            new_vertices = mrmeshnumpy.getNumpyVerts(meshlib_mesh)
            new_faces = mrmeshnumpy.getNumpyFaces(meshlib_mesh.topology)

            del meshlib_mesh
            gc.collect()
            
            cumesh.init(torch.from_numpy(new_vertices).float().to(coords.device), torch.from_numpy(new_faces).int().to(coords.device))
        
        # --- Branch: Bake On Vertices (skip UV unwrapping and texture creation) ---
        if bake_on_vertices:
            print('Baking colors on vertices...')
            out_vertices, out_faces = cumesh.read()
            out_vertices = out_vertices.cuda()
            out_faces = out_faces.cuda()
            cumesh.compute_vertex_normals()
            out_normals = cumesh.read_vertex_normals()
            
            # Map vertex positions back to original mesh for accurate attribute sampling
            # Use BVH to find the closest point on original mesh surface for more accurate colors
            _, face_id, uvw = bvh.unsigned_distance(out_vertices, return_uvw=True)
            orig_tri_verts = bvh.vertices[bvh.faces[face_id.long()]]  # (N_verts, 3, 3)
            mapped_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
            
            # Sample attributes at mapped positions from the voxel grid
            vertex_attrs = grid_sample_3d(
                attr_volume,
                torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
                shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
                grid=((mapped_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
                mode='trilinear',
            )
            
            # Extract base color and alpha per vertex (vertex_attrs shape: N_vertices x C)
            base_color_idx = attr_layout['base_color']
            alpha_idx = attr_layout['alpha']
            
            # Get RGB values and squeeze any extra dimensions to get (N, 3)
            vertex_colors_rgb = vertex_attrs[..., base_color_idx].cpu().numpy()
            vertex_colors_rgb = np.squeeze(vertex_colors_rgb)  # Remove batch dims if any
            if vertex_colors_rgb.ndim == 1:
                vertex_colors_rgb = vertex_colors_rgb[None, :]  # Ensure at least 2D
            vertex_colors_rgb = np.clip(vertex_colors_rgb * 255, 0, 255).astype(np.uint8)
            
            # Handle alpha based on texture_alpha_mode
            if texture_alpha_mode == "OPAQUE":
                # For OPAQUE mode, use full alpha (255)
                vertex_alpha = np.full((vertex_colors_rgb.shape[0], 1), 255, dtype=np.uint8)
            else:
                vertex_alpha = vertex_attrs[..., alpha_idx].cpu().numpy()
                vertex_alpha = np.squeeze(vertex_alpha)  # Remove batch dims if any
                vertex_alpha = np.clip(vertex_alpha * 255, 0, 255).astype(np.uint8)
                # Ensure alpha is 2D with shape (N, 1)
                if vertex_alpha.ndim == 1:
                    vertex_alpha = vertex_alpha[:, None]
            
            # Combine into RGBA
            vertex_colors_rgba = np.concatenate([vertex_colors_rgb, vertex_alpha], axis=-1)
            
            print("Finalizing mesh with vertex colors...")
            pbar.update(1)
            
            vertices_np = out_vertices.cpu().numpy()
            faces_np = out_faces.cpu().numpy()
            normals_np = out_normals.cpu().numpy()
            
            # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
            if reorient_vertices == '90 degrees':
                vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
                normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
            elif reorient_vertices == '-90 degrees':                
                vertices_np[:, 1], vertices_np[:, 2] = -vertices_np[:, 2].copy(), vertices_np[:, 1].copy()
                normals_np[:, 1], normals_np[:, 2] = -normals_np[:, 2].copy(), normals_np[:, 1].copy()                
            
            # Create mesh with vertex colors using ColorVisuals
            if use_custom_normals:
                textured_mesh = Trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_normals=normals_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )
            else:
                textured_mesh = Trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )                
            
            del cumesh
            gc.collect()
            
            # Return empty placeholder textures for vertex color mode
            placeholder_texture = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
            return (textured_mesh, placeholder_texture, placeholder_texture,)
        
        # --- Standard texture baking path ---
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
        pbar.update(1)
        
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
        orig_tri_verts = bvh.vertices[bvh.faces[face_id.long()]] # (N_new, 3, 3)
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)        
        
        torch.cuda.synchronize()
        
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
        pbar.update(1)
        
        mask = mask.cpu().numpy()
        
        # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
        base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha_mode = texture_alpha_mode
        
        # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
        if inpainting == 'telea':
            inpainting = cv2.INPAINT_TELEA
        else:
            inpainting = cv2.INPAINT_NS
        
        mask_inv = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask_inv, 3, inpainting)
        metallic = inpaint_channel(metallic, mask_inv, 1, inpainting)
        roughness = inpaint_channel(roughness, mask_inv, 1, inpainting)
        alpha = inpaint_channel(alpha, mask_inv, 1, inpainting)
        
        # Create PBR material
        # Standard PBR packs Metallic and Roughness into Blue and Green channels
        baseColorTexture_np = Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
        metallicRoughnessTexture_np = Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1))
        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=baseColorTexture_np,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=metallicRoughnessTexture_np,
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
        if reorient_vertices == '90 degrees':
            vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
            normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()            
        elif reorient_vertices == '-90 degrees':                
            vertices_np[:, 1], vertices_np[:, 2] = -vertices_np[:, 2].copy(), vertices_np[:, 1].copy()
            normals_np[:, 1], normals_np[:, 2] = -normals_np[:, 2].copy(), normals_np[:, 1].copy()          
        
        uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
        
        if use_custom_normals:
            textured_mesh = Trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                vertex_normals=normals_np,
                process=False,
                visual=Trimesh.visual.TextureVisuals(uv=uvs_np,material=material)
            )
        else:
            textured_mesh = Trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                process=False,
                visual=Trimesh.visual.TextureVisuals(uv=uvs_np,material=material)
            )
            
        pbar.update(1)        
        
        del cumesh
        gc.collect()         

        baseColorTexture = pil2tensor(baseColorTexture_np)
        metallicRoughnessTexture = pil2tensor(metallicRoughnessTexture_np)
        
        return (textured_mesh, baseColorTexture, metallicRoughnessTexture,)    

class Trellis2Remesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "remesh_project": ("FLOAT",{"default":0.0}),
                "dual_contouring_resolution": (["Auto","128","256","512","1024","2048"],{"default":"Auto"}),
                "remove_floaters": ("BOOLEAN",{"default":True}),
                "remove_inner_faces": ("BOOLEAN",{"default":False}),
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, remesh_band, remesh_project, dual_contouring_resolution, remove_floaters, remove_inner_faces):
        reset_cuda()
        
        mesh_copy = copy.deepcopy(mesh)
        
        if remove_floaters:
            mesh_copy = remove_floater(mesh_copy)
        
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        
        vertices = mesh_copy.vertices
        faces = mesh_copy.faces
        attr_volume = mesh_copy.attrs
        coords = mesh_copy.coords
        attr_layout = mesh_copy.layout
        voxel_size = mesh_copy.voxel_size        
        
        # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32, device='cuda')

        # Calculate grid dimensions based on AABB and voxel size                
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = [voxel_size, voxel_size, voxel_size]
            if isinstance(voxel_size, (list, tuple)):
                voxel_size = np.array(voxel_size)
            if isinstance(voxel_size, np.ndarray):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device='cuda')
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            if isinstance(grid_size, int):
                grid_size = [grid_size, grid_size, grid_size]
            if isinstance(grid_size, (list, tuple)):
                grid_size = np.array(grid_size)
            if isinstance(grid_size, np.ndarray):
                grid_size = torch.tensor(grid_size, dtype=torch.int32, device='cuda')
            voxel_size = (aabb[1] - aabb[0]) / grid_size

        # Move data to GPU
        vertices = vertices.cuda()
        faces = faces.cuda()
        
        # Initialize CUDA mesh handler
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        print(f"Current vertices: {cumesh.num_vertices}, faces: {cumesh.num_faces}")        
        
        vertices, faces = cumesh.read()
        
        del cumesh
        gc.collect()         
            
        # Build BVH for the current mesh to guide remeshing
        #print(f"Building BVH for current mesh...")
        #bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())
            
        print("Cleaning mesh...")        
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        
        if dual_contouring_resolution == "Auto":
            resolution = grid_size.max().item()
            print(f"Dual Contouring resolution: {resolution}")
        else:
            resolution = int(dual_contouring_resolution)
        
        print('Performing Dual Contouring ...')
        # Perform Dual Contouring remeshing (rebuilds topology)
        vertices, faces = CuMesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = scale * 1.1, # old calculation (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = True,
            remove_inner_faces = remove_inner_faces,
            #bvh = bvh,
        )
        
        if remove_floaters:
            vertices, faces = remove_floater2(vertices.cpu().numpy(),faces.cpu().numpy())
            vertices = torch.from_numpy(vertices).contiguous().float()
            faces = torch.from_numpy(faces).contiguous().int() 
            
        print(f"After remeshing: {len(vertices)} vertices, {len(faces)} faces")                                 
        
        mesh_copy.vertices = vertices.to(mesh_copy.device)
        mesh_copy.faces = faces.to(mesh_copy.device) 
                
        return (mesh_copy,)
        
class Trellis2ReconstructMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "resolution": ([128,256,512,1024,2048],{"default":512}),             
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, remesh_band, resolution):
        reset_cuda()
        
        mesh_copy = copy.deepcopy(mesh)
        
        vertices = mesh_copy.vertices.cuda()
        faces = mesh_copy.faces.cuda()
        
        # Perform Dual Contouring remeshing (rebuilds topology)
        print('Reconstructing mesh ...')
        vertices, faces = CuMesh.remeshing.reconstruct_mesh_dc(vertices, faces, resolution, verbose=True)
        
        print(f"After reconstruction: {len(vertices)} vertices, {len(faces)} faces")                                 
        
        mesh_copy.vertices = vertices.to(mesh_copy.device)
        mesh_copy.faces = faces.to(mesh_copy.device) 
                
        return (mesh_copy,)   

class Trellis2ReconstructMeshWithQuad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "resolution": ([128,256,512,1024,2048],{"default":512}),
                "remove_floaters": ("BOOLEAN",{"default":True}),
                "remove_inner_faces": ("BOOLEAN",{"default":False}),
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, remesh_band, resolution, remove_floaters, remove_inner_faces):
        reset_cuda()
        
        mesh_copy = copy.deepcopy(mesh)
        
        vertices = mesh_copy.vertices.cuda()
        faces = mesh_copy.faces.cuda()
        
        # Perform Dual Contouring remeshing (rebuilds topology)
        print('Reconstructing mesh ...')
        vertices, faces = CuMesh.remeshing.reconstruct_mesh_dc_quad(vertices, faces, resolution, verbose=True, remove_inner_faces = remove_inner_faces)
        
        if remove_floaters:
            vertices, faces = remove_floater2(vertices.cpu().numpy(),faces.cpu().numpy())
            vertices = torch.from_numpy(vertices).contiguous().float()
            faces = torch.from_numpy(faces).contiguous().int()         
        
        print(f"After reconstruction: {len(vertices)} vertices, {len(faces)} faces")                                 
        
        mesh_copy.vertices = vertices.to(mesh_copy.device)
        mesh_copy.faces = faces.to(mesh_copy.device) 
                
        return (mesh_copy,)       
        
class Trellis2MeshTexturing:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "trimesh": ("TRIMESH",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":3.00,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.20,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":3.00,"min":0.00,"max":9.99,"step":0.01}), 
                "resolution": ([512,1024,1536],{"default":1024}),
                "texture_size": ("INT",{"default":4096,"min":512,"max":16384}),
                "texture_alpha_mode": (["OPAQUE","MASK","BLEND"],{"default":"OPAQUE"}),
                "double_side_material": ("BOOLEAN",{"default":False}), 
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "bake_on_vertices": ("BOOLEAN",{"default":False}),
                "use_custom_normals": ("BOOLEAN",{"default":False}),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60.0,"min":0.0,"max":359.9}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "inpainting": (["telea","ns"],{"default":"telea"}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),                
            },           
        }

    RETURN_TYPES = ("TRIMESH","IMAGE","IMAGE",)
    RETURN_NAMES = ("trimesh","base_color_texture","metallic_roughness_texture",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, trimesh, seed, texture_steps, texture_guidance_strength, texture_guidance_rescale, texture_rescale_t, resolution, texture_size, texture_alpha_mode, double_side_material, texture_guidance_interval_start, texture_guidance_interval_end, max_views,bake_on_vertices,use_custom_normals,mesh_cluster_threshold_cone_half_angle_rad, sampler, inpainting,    
        verbose, dino_lock, dino_substeps, dino_foundation_cap, moge_camera_config = None):
            
        if pipeline.isPixal3D:
            raise Exception('Pixal3D does not support Mesh Texturing')
        
        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images

        #image = tensor2pil(image)
        
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]                
        
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}

        textured_mesh, baseColorTexture_np, metallicRoughnessTexture_np = pipeline.texture_mesh(mesh=trimesh, 
            image=image_in, 
            seed=seed, 
            tex_slat_sampler_params = tex_slat_sampler_params,
            resolution = resolution,
            texture_size = texture_size,
            texture_alpha_mode = texture_alpha_mode,
            double_side_material = double_side_material,
            max_views = max_views,
            bake_on_vertices = bake_on_vertices,
            use_custom_normals = use_custom_normals,
            mesh_cluster_threshold_cone_half_angle_rad = mesh_cluster_threshold_cone_half_angle_rad,
            sampler = sampler,
            inpainting = inpainting,
            verbose = verbose,
            dino_lock = dino_lock,
            dino_substeps = dino_substeps,
            dino_foundation_cap = dino_foundation_cap
        )            

        baseColorTexture = pil2tensor(baseColorTexture_np)
        metallicRoughnessTexture = pil2tensor(metallicRoughnessTexture_np)
        
        return (textured_mesh, baseColorTexture, metallicRoughnessTexture, )
        
class Trellis2MeshTexturingMultiView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "front_image": ("IMAGE",),
                "trimesh": ("TRIMESH",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":3.00,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.20,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":3.00,"min":0.00,"max":9.99,"step":0.01}), 
                "resolution": ([512,1024,1536],{"default":1024}),
                "texture_size": ("INT",{"default":4096,"min":512,"max":16384}),
                "texture_alpha_mode": (["OPAQUE","MASK","BLEND"],{"default":"OPAQUE"}),
                "double_side_material": ("BOOLEAN",{"default":False}), 
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "bake_on_vertices": ("BOOLEAN",{"default":False}),
                "use_custom_normals": ("BOOLEAN",{"default":False}),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60.0,"min":0.0,"max":359.9}),
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "inpainting": (["telea","ns"],{"default":"telea"}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}), 
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
            },
            "optional": {
                "back_image": ("IMAGE",),
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),                
            }
        }

    RETURN_TYPES = ("TRIMESH","IMAGE","IMAGE",)
    RETURN_NAMES = ("trimesh","base_color_texture","metallic_roughness_texture",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, 
        pipeline, 
        front_image, 
        trimesh, 
        seed, 
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale, 
        texture_rescale_t, 
        resolution, 
        texture_size, 
        texture_alpha_mode, 
        double_side_material, 
        texture_guidance_interval_start, 
        texture_guidance_interval_end, 
        bake_on_vertices,
        use_custom_normals,
        mesh_cluster_threshold_cone_half_angle_rad,
        front_axis,
        blend_temperature,
        sampler,
        inpainting,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        back_image = None,
        left_image = None,
        right_image = None):
        
        reset_cuda()
        
        # Convert front image tensor to PIL
        front_pil = tensor2pil(front_image)
        
        # Convert optional view image tensors to PIL
        back_pil = tensor2pil(back_image) if back_image is not None else None
        left_pil = tensor2pil(left_image) if left_image is not None else None
        right_pil = tensor2pil(right_image) if right_image is not None else None        
        
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]                
        
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}

        textured_mesh, baseColorTexture_np, metallicRoughnessTexture_np = pipeline.texture_mesh_multiview(mesh=trimesh, 
            front=front_pil,
            back=back_pil,
            left=left_pil,
            right=right_pil,
            seed=seed, 
            tex_slat_sampler_params = tex_slat_sampler_params,
            resolution = resolution,
            texture_size = texture_size,
            texture_alpha_mode = texture_alpha_mode,
            double_side_material = double_side_material,
            bake_on_vertices = bake_on_vertices,
            use_custom_normals = use_custom_normals,
            mesh_cluster_threshold_cone_half_angle_rad = mesh_cluster_threshold_cone_half_angle_rad,
            front_axis = front_axis,
            blend_temperature = blend_temperature,
            sampler = sampler,
            inpainting = inpainting,
            verbose = verbose,
            dino_lock = dino_lock,
            dino_substeps = dino_substeps,
            dino_foundation_cap = dino_foundation_cap
        )            

        baseColorTexture = pil2tensor(baseColorTexture_np)
        metallicRoughnessTexture = pil2tensor(metallicRoughnessTexture_np)
        
        return (textured_mesh, baseColorTexture, metallicRoughnessTexture, )        
        
class Trellis2LoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "The glb path with mesh to load."}), 
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    OUTPUT_TOOLTIPS = ("The glb model with mesh to texturize.",)
    
    FUNCTION = "load"
    CATEGORY = "Trellis2Wrapper"
    DESCRIPTION = "Loads a glb model from the given path."

    def load(self, glb_path):
        if not os.path.exists(glb_path):
            glb_path = os.path.join(folder_paths.get_input_directory(), glb_path)
        
        trimesh = Trimesh.load(glb_path, force="mesh")
        
        return (trimesh,)  
        
class Trellis2PreProcessImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding": ("INT",{"default":0,"min":0,"max":1024}),
                "remove_background": ("BOOLEAN",{"default":False}),
                "max_size": ("INT",{"default":2048,"min":512,"max":8192,"step":128}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def process(self, image, padding, remove_background, max_size):
        if image.ndim == 3:
            image = tensor2pil(image)
            
            if remove_background:
                from rembg import remove
                image = remove(image)
            
            image = self.preprocess_image(image, max_size)
            
            if padding>0:
                border = (int(padding), int(padding), int(padding), int(padding))
                fill_color = self.parse_fill_for_image("0,0,0,255", image)
                image = ImageOps.expand(image,border=border,fill=fill_color)
            
            image = pil2tensor(image)
        elif image.ndim == 4:
            images = convert_tensor_images_to_pil(image)
            tensor_list = []
            for img in images:
                if remove_background:
                    from rembg import remove
                    img = remove(img)
                
                img = self.preprocess_image(img, max_size)
                
                if padding>0:
                    border = (int(padding), int(padding), int(padding), int(padding))
                    fill_color = self.parse_fill_for_image("0,0,0,255", img)
                    img = ImageOps.expand(img,border=border,fill=fill_color)
                
                tensor_list.append(pil2tensor(img))
                
                max_h = max(t.shape[-3] for t in tensor_list)
                max_w = max(t.shape[-2] for t in tensor_list)

                resized_tensors = []

                for t in tensor_list:
                    # Ensure tensor is [C, H, W] for PyTorch's interpolate function
                    # Current shape is likely [H, W, C] or [1, H, W, C]
                    temp_t = t.squeeze() # Get to [H, W, C]
                    temp_t = temp_t.permute(2, 0, 1).unsqueeze(0) # Becomes [1, C, H, W]
                    
                    # 2. Resize to the max dimensions
                    # Using 'bicubic' or 'bilinear' for better quality than 'nearest'
                    temp_t = F.interpolate(temp_t, size=(max_h, max_w), mode='bicubic', align_corners=False)
                    
                    # 3. Convert back to ComfyUI format [H, W, C]
                    temp_t = temp_t.squeeze(0).permute(1, 2, 0)
                    resized_tensors.append(temp_t)                
                
            image = torch.stack(resized_tensors)
        
        return (image,)    

    def parse_fill_for_image(self, fill: str, img):
        values = [int(x.strip()) for x in fill.split(",")]

        if img.mode in ("L", "P"):
            return values[0]

        if img.mode == "RGB":
            return tuple(values[:3])

        if img.mode == "RGBA":
            return tuple(values[:4])

        raise ValueError(f"Unsupported image mode: {img.mode}")         


    def preprocess_image(self, input: Image.Image, max_res) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, max_res / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        # if has_alpha:
            # output = input
        # else:
            # input = input.convert('RGB')
            # if self.low_vram:
                # self.rembg_model.to(self.device)
            # output = self.rembg_model(input)
            # if self.low_vram:
                # self.rembg_model.cpu()
        output = input
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output    

class Trellis2MeshRefiner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "trimesh": ("TRIMESH",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0x7fffffff}),
                "resolution": ([512,1024,1536],{"default":1024}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                  
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":3.00,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.20,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":3.00,"min":0.00,"max":9.99,"step":0.01}),               
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "generate_texture_slat": ("BOOLEAN", {"default":True}),
                "downsampling":([16,32,64],{"default":16}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", "BVH", )
    RETURN_NAMES = ("mesh", "bvh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, trimesh, image, seed, resolution,
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,        
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale,
        texture_rescale_t,        
        max_num_tokens,
        generate_texture_slat,
        downsampling,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        use_tiled_decoder,
        max_views,
        sampler,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap):

        reset_cuda()

        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images
        
        shape_guidance_interval = [shape_guidance_interval_start,shape_guidance_interval_end]
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]        
        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}       
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
        
        mesh = pipeline.refine_mesh(mesh = trimesh, 
                                    image=image_in, 
                                    seed=seed, 
                                    shape_slat_sampler_params = shape_slat_sampler_params, 
                                    tex_slat_sampler_params = tex_slat_sampler_params, 
                                    resolution = resolution, 
                                    max_num_tokens = max_num_tokens, 
                                    generate_texture_slat=generate_texture_slat, 
                                    downsampling=downsampling, 
                                    use_tiled=use_tiled_decoder, 
                                    max_views = max_views, 
                                    sampler = sampler,
                                    verbose = verbose,
                                    dino_lock = dino_lock,
                                    dino_substeps = dino_substeps,
                                    dino_foundation_cap = dino_foundation_cap)[0]         
        
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()        
       
        # Build BVH for the current mesh to guide remeshing
        if generate_texture_slat:
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())           
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()
        else:
            print('Not building BVH, only used for texturing')
            bvh = None
        
        return (mesh, bvh,)        

class Trellis2PostProcess2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "fill_holes": ("BOOLEAN", {"default":True}),
                "fix_normals": ("BOOLEAN", {"default":False}),
                "fix_face_orientation": ("BOOLEAN", {"default":True}),
                "remove_duplicate_faces": ("BOOLEAN",{"default":True}),
                "weld_vertices": ("BOOLEAN",{"default":True}),
                "weld_vertices_digits": ("INT",{"default":4,"min":1,"max":8}),
                "smooth": ("BOOLEAN",{"default":False}),
                "smooth_iterations": ("INT",{"default":10,"min":1,"max":99,"step":1}),
                "subdivide": ("BOOLEAN",{"default":False}),
                "subdivide_iterations": ("INT",{"default":1,"min":1,"max":10}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, fill_holes, fix_normals, fix_face_orientation, remove_duplicate_faces, weld_vertices, weld_vertices_digits,smooth,smooth_iterations,subdivide,subdivide_iterations):
        mesh_copy = copy.deepcopy(mesh)
        
        vertices_np = mesh_copy.vertices.cpu().numpy()
        faces_np = mesh_copy.faces.cpu().numpy()
        
        trimesh = Trimesh.Trimesh(vertices=vertices_np,faces=faces_np)
        
        print(f"Initial mesh: {len(trimesh.faces)} faces")
        #print(f"Is winding consistent? {trimesh.is_winding_consistent}")        
        
        if fix_normals:
            print('Fixing normals ...')
            trimesh.fix_normals()       
            
        if fix_face_orientation:
            if trimesh.is_watertight:
                print('Mesh is watertight, fixing inversion ...')
                Trimesh.repair.fix_inversion(trimesh)
            else:
                print('Mesh is not watertight, cannot fix inversion')

        if remove_duplicate_faces:
            print('Removing duplicate faces ...')
            trimesh.update_faces(trimesh.unique_faces()) 
        
        if fill_holes:
            print('Filling holes ...')
            trimesh.fill_holes()     

        if weld_vertices:
            vertices_count = len(trimesh.vertices)
            trimesh.merge_vertices(digits_vertex=weld_vertices_digits)
            new_vertices_count = len(trimesh.vertices)
            nb_vertices_removed = vertices_count - new_vertices_count            
            faces_count = len(trimesh.faces)
            trimesh.remove_unreferenced_vertices()
            trimesh.update_faces(trimesh.nondegenerate_faces())
            new_faces_count = len(trimesh.faces)
            nb_faces_removed = faces_count - new_faces_count
            print(f"Weld Vertices: Removed {nb_vertices_removed} vertices / {nb_faces_removed} faces")
        
        if smooth:
            print('Smoothing ...')
            Trimesh.smoothing.filter_taubin(trimesh, lamb=0.5, nu=-0.53, iterations=smooth_iterations)
            
        if subdivide:
            print('Subdividing ...')
            trimesh = trimesh.subdivide_loop(iterations=subdivide_iterations)
        
        new_vertices = torch.from_numpy(trimesh.vertices).float()
        new_faces = torch.from_numpy(trimesh.faces).int()

        print(f"After postprocessing: {len(new_faces)} faces")
        
        mesh_copy.vertices = new_vertices.to(mesh_copy.device)
        mesh_copy.faces = new_faces.to(mesh_copy.device) 
        
        del trimesh
        gc.collect()
                
        return (mesh_copy,)    
        
class Trellis2SmoothMeshWithPyMeshlab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "isotropic_remeshing_iterations": ("INT",{"default":5,"min":1,"max":99}),
                "isotropic_remeshing_target_len": ("FLOAT",{"default":0.25,"min":0.10,"max":0.90,"step":0.01}),
                "taubin_smoothing_step": ("INT",{"default":30,"min":1,"max":200}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, isotropic_remeshing_iterations, isotropic_remeshing_target_len, taubin_smoothing_step):
        mesh_copy = copy.deepcopy(mesh)
        
        vertices_np = mesh_copy.vertices.cpu().numpy()
        faces_np = mesh_copy.faces.cpu().numpy()

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertices_np, faces_np))
        #ms.apply_filter('generate_surface_reconstruction_screened_poisson', depth=9)
        ms.apply_filter('meshing_isotropic_explicit_remeshing',
                         targetlen=pymeshlab.PercentageValue(isotropic_remeshing_target_len),  # tune relative to model size
                         iterations=isotropic_remeshing_iterations)
        ms.apply_filter('apply_coord_taubin_smoothing', lambda_=0.5, mu=-0.53, stepsmoothnum=taubin_smoothing_step, selected=False)
        result = ms.current_mesh()
        
        new_vertices = torch.from_numpy(result.vertex_matrix()).float()
        new_faces = torch.from_numpy(result.face_matrix()).int()
        
        mesh_copy.vertices = new_vertices.to(mesh_copy.device)
        mesh_copy.faces = new_faces.to(mesh_copy.device) 
        
        del ms
        gc.collect()
                
        return (mesh_copy,)    

class Trellis2SmoothTrimeshWithPyMeshlab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "isotropic_remeshing_iterations": ("INT",{"default":5,"min":1,"max":99}),
                "isotropic_remeshing_target_len": ("FLOAT",{"default":0.25,"min":0.10,"max":0.90,"step":0.01}),
                "taubin_smoothing_step": ("INT",{"default":30,"min":1,"max":200}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, isotropic_remeshing_iterations, isotropic_remeshing_target_len, taubin_smoothing_step):
        mesh_copy = trimesh.copy()       

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh_copy.vertices, mesh_copy.faces))
        #ms.apply_filter('generate_surface_reconstruction_screened_poisson', depth=9)
        ms.apply_filter('meshing_isotropic_explicit_remeshing',
                         targetlen=pymeshlab.PercentageValue(isotropic_remeshing_target_len),  # tune relative to model size
                         iterations=isotropic_remeshing_iterations)
        ms.apply_filter('apply_coord_taubin_smoothing', lambda_=0.5, mu=-0.53, stepsmoothnum=taubin_smoothing_step, selected=False)
        result = ms.current_mesh()
        
        mesh_copy = Trimesh.Trimesh(result.vertex_matrix(), result.face_matrix())
        
        del ms
        gc.collect()
                
        return (mesh_copy,)           

class Trellis2OvoxelExportToGLB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "resolution": ([512,1024],{"default":1024}),
                "texture_size": ([512,1024,2048,4096],{"default":2048}),
                "target_face_num": ("INT",{"default":2000000,"min":500,"max":16000000}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, resolution, texture_size, target_face_num):
        mesh_copy = copy.deepcopy(mesh)

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh_copy.vertices,
            faces=mesh_copy.faces,
            attr_volume=mesh_copy.attrs,
            coords=mesh_copy.coords,
            attr_layout=mesh_copy.layout,
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=target_face_num,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )
                
        return (glb,)

class Trellis2TrimeshToMeshWithVoxel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "resolution": ([512,1024],{"default":1024}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, resolution):       
        mesh_copy = trimesh.copy()
        
        mvoxel = self.get_voxelmesh_from_trimesh(mesh_copy, resolution)
        
        return (mvoxel,)        
        
    def get_voxelmesh_from_trimesh(self, mesh, resolution):
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
        
        coords = torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)                
        coords = coords.cpu()

        del voxel_indices
        del dual_vertices
        del intersected
        gc.collect()
            
        pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }

        mvoxel = MeshWithVoxel(
                    vertices, faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = coords,
                    attrs = None,
                    voxel_shape = None,
                    layout=pbr_attr_layout
                    )
                    
        return mvoxel
        
class Trellis2Continue:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
                "input_2": (any,),
            },
        }

    RETURN_TYPES = (any, any, )
    RETURN_NAMES = ("output_1", "output_2", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1, input_2):        
        return (input_1, input_2,)
        
class Trellis2Continue3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
                "input_2": (any,),
                "input_3": (any,),
            },
        }

    RETURN_TYPES = (any, any, any)
    RETURN_NAMES = ("output_1", "output_2", "output_3")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1, input_2, input_3):        
        return (input_1, input_2, input_3)      

class Trellis2Continue4:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
                "input_2": (any,),
                "input_3": (any,),
                "input_4": (any,),
            },
        }

    RETURN_TYPES = (any, any, any, any)
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1, input_2, input_3, input_4):        
        return (input_1, input_2, input_3, input_4)    

class Trellis2Continue5:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
                "input_2": (any,),
                "input_3": (any,),
                "input_4": (any,),
                "input_5": (any,),
            },
        }

    RETURN_TYPES = (any, any, any, any, any,)
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4", "output_5")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1, input_2, input_3, input_4, input_5):        
        return (input_1, input_2, input_3, input_4, input_5)   

class Trellis2Continue6:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
                "input_2": (any,),
                "input_3": (any,),
                "input_4": (any,),
                "input_5": (any,),
                "input_6": (any,),
            },
        }

    RETURN_TYPES = (any, any, any, any, any, any)
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4", "output_5", "output_6")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1, input_2, input_3, input_4, input_5, input_6):        
        return (input_1, input_2, input_3, input_4, input_5, input_6)         
        
class Trellis2MeshWithVoxelToMeshlibMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
            },
        }

    RETURN_TYPES = ("MESHLIB_MESH", )
    RETURN_NAMES = ("meshlib_mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh):        
        meshlib_mesh = mrmeshnumpy.meshFromFacesVerts(mesh.faces.cpu().numpy(), mesh.vertices.cpu().numpy())                                 
        return (meshlib_mesh,)

class Trellis2FillHolesWithMeshlib:
    """Fill all holes in a mesh"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
            },
        }
    
    RETURN_TYPES = ("MESHWITHVOXEL", "INT")
    RETURN_NAMES = ("mesh", "holes_filled")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    DESCRIPTION = "Fill all holes in a mesh using optimal triangulation."

    def process(self, mesh):
        import meshlib.mrmeshpy as mrmeshpy
        
        mesh_copy = copy.deepcopy(mesh)
        mesh = mrmeshnumpy.meshFromFacesVerts(mesh_copy.faces.detach().clone().cpu().numpy(), mesh_copy.vertices.detach().clone().cpu().numpy())
        
        hole_edges = mesh.topology.findHoleRepresentiveEdges()
        holes_filled = 0
        
        nb_holes = len(hole_edges)
        print(f"{nb_holes} holes found")

        if nb_holes > 0:
            progress_bar = tqdm(total=nb_holes, desc="Filling holes")
            pbar = ProgressBar(nb_holes)
            
            last_reported_percent = -1  # Initialize at -1 to ensure 0% triggers an update
            
            for i, e in enumerate(hole_edges):                
                params = mrmeshpy.FillHoleParams()
                params.metric = mrmeshpy.getUniversalMetric(mesh)
                mrmeshpy.fillHole(mesh, e, params)
                
                # Calculate current progress
                current_step = i + 1
                current_percent = int((current_step / nb_holes) * 100)
                
                # Only update the UI if the percentage has moved up
                if current_percent > last_reported_percent:
                    # Calculate how many holes have been filled since the last UI update
                    # This handles cases where 1% might represent multiple holes
                    if last_reported_percent == -1:
                        # First update
                        progress_bar.update(current_step)
                        pbar.update(current_step)
                    else:
                        # Update by the difference since the last check
                        last_step = int((last_reported_percent * nb_holes) / 100)
                        diff = current_step - last_step
                        progress_bar.update(diff)
                        pbar.update(diff)
                    
                    last_reported_percent = current_percent
                        
            progress_bar.close()            
        
        new_vertices = mrmeshnumpy.getNumpyVerts(mesh)
        new_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)

        del mesh
        gc.collect()
        
        mesh_copy.vertices = torch.from_numpy(new_vertices).float().to(mesh_copy.device)
        mesh_copy.faces = torch.from_numpy(new_faces).int().to(mesh_copy.device)
        
        return (mesh_copy, holes_filled) 
        
class Trellis2FillHolesNicelyWithMeshlib:
    """Fill all holes in a mesh"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
            },
        }
    
    RETURN_TYPES = ("MESHWITHVOXEL", "INT")
    RETURN_NAMES = ("mesh", "holes_filled")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    DESCRIPTION = "Fill all holes in a mesh using optimal triangulation."

    def process(self, mesh):
        import meshlib.mrmeshpy as mrmeshpy
        
        mesh_copy = copy.deepcopy(mesh)
        mesh = mrmeshnumpy.meshFromFacesVerts(mesh_copy.faces.detach().clone().cpu().numpy(), mesh_copy.vertices.detach().clone().cpu().numpy())
        
        hole_edges = mesh.topology.findHoleRepresentiveEdges()
        holes_filled = 0
        
        nb_holes = len(hole_edges)
        print(f"{nb_holes} holes found")

        if nb_holes > 0:
            progress_bar = tqdm(total=nb_holes, desc="Filling holes")
            pbar = ProgressBar(nb_holes)
            
            last_reported_percent = -1  # Initialize at -1 to ensure 0% triggers an update
            
            for i, e in enumerate(hole_edges):          
                params = mrmeshpy.FillHoleNicelySettings()
                params.triangulateParams.metric = mrmeshpy.getMinAreaMetric(mesh)
                params.smoothCurvature = False
                #  Fill hole represented by `e`
                mrmeshpy.fillHoleNicely(mesh, e, params)
                
                # Calculate current progress
                current_step = i + 1
                current_percent = int((current_step / nb_holes) * 100)
                
                # Only update the UI if the percentage has moved up
                if current_percent > last_reported_percent:
                    # Calculate how many holes have been filled since the last UI update
                    # This handles cases where 1% might represent multiple holes
                    if last_reported_percent == -1:
                        # First update
                        progress_bar.update(current_step)
                        pbar.update(current_step)
                    else:
                        # Update by the difference since the last check
                        last_step = int((last_reported_percent * nb_holes) / 100)
                        diff = current_step - last_step
                        progress_bar.update(diff)
                        pbar.update(diff)
                    
                    last_reported_percent = current_percent
                        
            progress_bar.close()            
        
        new_vertices = mrmeshnumpy.getNumpyVerts(mesh)
        new_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)

        del mesh
        gc.collect()
        
        mesh_copy.vertices = torch.from_numpy(new_vertices).float().to(mesh_copy.device)
        mesh_copy.faces = torch.from_numpy(new_faces).int().to(mesh_copy.device)
        
        return (mesh_copy, holes_filled)         
        
class Trellis2SmoothNormals:    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }
    
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def process(self, trimesh):
        new_mesh = trimesh.copy()
        new_mesh.vertex_normals = Trimesh.smoothing.get_vertices_normals(new_mesh)
        
        return (new_mesh,)         

class Trellis2RemeshWithQuad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "remesh_project": ("FLOAT",{"default":0.0}),
                "dual_contouring_resolution": (["Auto","128","256","512","1024","2048"],{"default":"Auto"}),
                "remove_floaters": ("BOOLEAN",{"default":True}),
                "remove_inner_faces": ("BOOLEAN",{"default":True})
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, remesh_band, remesh_project, dual_contouring_resolution, remove_floaters, remove_inner_faces):
        reset_cuda()
        
        mesh_copy = copy.deepcopy(mesh)
        
        if remove_floaters:
            mesh_copy = remove_floater(mesh_copy)
        
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        
        vertices = mesh_copy.vertices
        faces = mesh_copy.faces
        attr_volume = mesh_copy.attrs
        coords = mesh_copy.coords
        attr_layout = mesh_copy.layout
        voxel_size = mesh_copy.voxel_size        
        
        # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32, device='cuda')

        # Calculate grid dimensions based on AABB and voxel size                
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = [voxel_size, voxel_size, voxel_size]
            if isinstance(voxel_size, (list, tuple)):
                voxel_size = np.array(voxel_size)
            if isinstance(voxel_size, np.ndarray):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device='cuda')
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            if isinstance(grid_size, int):
                grid_size = [grid_size, grid_size, grid_size]
            if isinstance(grid_size, (list, tuple)):
                grid_size = np.array(grid_size)
            if isinstance(grid_size, np.ndarray):
                grid_size = torch.tensor(grid_size, dtype=torch.int32, device='cuda')
            voxel_size = (aabb[1] - aabb[0]) / grid_size

        # Move data to GPU
        vertices = vertices.cuda()
        faces = faces.cuda()
        
        # Initialize CUDA mesh handler
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        print(f"Current vertices: {cumesh.num_vertices}, faces: {cumesh.num_faces}")
        
        vertices, faces = cumesh.read()
        
        del cumesh
        gc.collect()         
            
        # Build BVH for the current mesh to guide remeshing
        #print(f"Building BVH for current mesh...")
        #bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())
            
        print("Cleaning mesh...")        
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        
        if dual_contouring_resolution == "Auto":
            resolution = grid_size.max().item()
            print(f"Dual Contouring resolution: {resolution}")
        else:
            resolution = int(dual_contouring_resolution)
        
        print('Performing Dual Contouring ...')
        # Perform Dual Contouring remeshing (rebuilds topology)
        vertices, faces = CuMesh.remeshing.remesh_narrow_band_dc_quad(
            vertices, faces,
            center = center,
            scale = scale * 1.1, # old calculation (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = True,
            remove_inner_faces = remove_inner_faces,
            #bvh = bvh,
        )
        
        if remove_floaters:
            vertices, faces = remove_floater2(vertices.cpu().numpy(),faces.cpu().numpy())
            vertices = torch.from_numpy(vertices).contiguous().float()
            faces = torch.from_numpy(faces).contiguous().int() 
            
        print(f"After remeshing: {len(vertices)} vertices, {len(faces)} faces")                                 
        
        mesh_copy.vertices = vertices.to(mesh_copy.device)
        mesh_copy.faces = faces.to(mesh_copy.device) 
                
        return (mesh_copy,)   

class Trellis2BatchSimplifyMeshAndExport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "target_face_num": ("STRING",{"default":"2000000,1000000,500000,100000,50000,10000,5000,2500,1000"}),
                "method": (["Cumesh","Meshlib"],{"default":"Cumesh"}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "reorient_vertices":(["None","90 degrees","-90 degrees"],{"default":"90 degrees"}),
                "filename_prefix":("STRING",),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
                "weld_vertices": ("BOOLEAN",{"default":True}),
                "weld_vertices_digits":("INT",{"default":4,"min":1,"max":8}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("lst_glb_path", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, target_face_num, method, fill_holes, reorient_vertices, filename_prefix, file_format, weld_vertices, weld_vertices_digits):
        lst_output_mesh = []
        list_of_faces = parse_string_to_int_list(target_face_num)
        if len(list_of_faces)>0:
            cumesh = CuMesh.CuMesh()
            mesh_copy = copy.deepcopy(mesh)
            
            for target_nbfaces in list_of_faces:
                print(f"Processing at {target_nbfaces} ...")                
                
                vertices = mesh_copy.vertices.detach().clone().cpu().numpy()
                faces = mesh_copy.faces.detach().clone().cpu().numpy()                                
                
                if method=="Cumesh":
                    cumesh.init(torch.from_numpy(vertices).float().cuda(), torch.from_numpy(faces).int().cuda())
                    # options = {
                        # 'method': 'legacy'
                    # }                       
                    cumesh.simplify(target_nbfaces, verbose=True)
                    vertices, faces = cumesh.read()
                    vertices = vertices.cpu().numpy()
                    faces = faces.cpu().numpy()
                elif method=="Meshlib":
                    vertices, faces = simplify_with_meshlib(vertices, faces, target_nbfaces)
                else:
                    raise Exception("Unknown simplification method")
                
                if fill_holes:
                    import meshlib.mrmeshpy as mrmeshpy

                    mmesh = mrmeshnumpy.meshFromFacesVerts(faces, vertices)
                    
                    hole_edges = mmesh.topology.findHoleRepresentiveEdges()
                    
                    nb_holes = len(hole_edges)
                    print(f"{nb_holes} holes found")

                    if nb_holes > 0:
                        progress_bar = tqdm(total=nb_holes, desc="Filling holes")
                        
                        last_reported_percent = -1  # Initialize at -1 to ensure 0% triggers an update
                        
                        for i, e in enumerate(hole_edges):
                            params = mrmeshpy.FillHoleParams()
                            params.metric = mrmeshpy.getUniversalMetric(mmesh)
                            mrmeshpy.fillHole(mmesh, e, params)
                            
                            # Calculate current progress
                            current_step = i + 1
                            current_percent = int((current_step / nb_holes) * 100)
                            
                            # Only update the UI if the percentage has moved up
                            if current_percent > last_reported_percent:
                                # Calculate how many holes have been filled since the last UI update
                                # This handles cases where 1% might represent multiple holes
                                if last_reported_percent == -1:
                                    # First update
                                    progress_bar.update(current_step)
                                else:
                                    # Update by the difference since the last check
                                    last_step = int((last_reported_percent * nb_holes) / 100)
                                    diff = current_step - last_step
                                    progress_bar.update(diff)
                                
                                last_reported_percent = current_percent
                                    
                        progress_bar.close()                         
                    
                    vertices = mrmeshnumpy.getNumpyVerts(mmesh)
                    faces = mrmeshnumpy.getNumpyFaces(mmesh.topology)

                    del mmesh
                    gc.collect()
                
                if reorient_vertices == '90 degrees':
                    vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
                elif reorient_vertices == '-90 degrees':
                    vertices[:, 1], vertices[:, 2] = -vertices[:, 2], vertices[:, 1]
                
                trimesh = Trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    process=False
                )
                
                if weld_vertices:
                    vertices_count = len(trimesh.vertices)
                    trimesh.merge_vertices(digits_vertex=digits)
                    new_vertices_count = len(trimesh.vertices)
                    nb_vertices_removed = vertices_count - new_vertices_count
                    print(f"Weld Vertices: Removed {nb_vertices_removed} vertices")                    
                    

                filename_prefix_with_nbfaces = f"{filename_prefix}_{target_nbfaces}"

                full_output_folder, filename, counter, subfolder, filename_prefix_with_nbfaces = folder_paths.get_save_image_path(filename_prefix_with_nbfaces, folder_paths.get_output_directory())                
                output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
                output_glb_path.parent.mkdir(exist_ok=True)
                
                trimesh.export(output_glb_path, file_type=file_format)
                
                lst_output_mesh.append(str(output_glb_path))

                del trimesh
            
            del cumesh
            del mesh_copy
        
        return (lst_output_mesh,)   

class Trellis2WeldVertices:    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "merge_texture": ("BOOLEAN",{"default":True}),
                "merge_normals": ("BOOLEAN",{"default":True}),
                "digits":("INT",{"default":4,"min":1,"max":8}),
            },
        }
    
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def process(self, trimesh, merge_texture, merge_normals, digits):
        new_mesh = trimesh.copy()
        vertices_count = len(new_mesh.vertices)
        new_mesh.merge_vertices(merge_tex=merge_texture, merge_norm=merge_normals, digits_vertex=digits, digits_norm=digits, digits_uv=digits)
        new_vertices_count = len(new_mesh.vertices)
        nb_vertices_removed = vertices_count - new_vertices_count
        faces_count = len(new_mesh.faces)
        new_mesh.remove_unreferenced_vertices()
        new_mesh.update_faces(new_mesh.nondegenerate_faces())
        new_faces_count = len(new_mesh.faces)
        nb_faces_removed = faces_count - new_faces_count
        print(f"Weld Vertices: Removed {nb_vertices_removed} vertices / {nb_faces_removed} faces")
        
        return (new_mesh,)   

class Trellis2StringSelector:    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strings": ("STRING",),
                "index": ("INT",{"default":0,"min":0,"max":1000}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def process(self, strings, index):
        if isinstance(strings, list):
            if len(strings) == 0:
                string = ""
            elif len(strings)<=index:
                index = len(strings)-1
                
            string = strings[index]
        elif isinstance(strings, str):
            string = strings
        else:
            raise Exception("string must be a list of a string")            
        
        return (string,)

class Trellis2FillHolesWithCuMesh:
    """Fill all holes in a mesh"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "max_permieters": ("FLOAT",{"default":0.030,"min":0.001,"max":99.999,"step":0.001}),
            },
        }
    
    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def process(self, mesh, max_permieters):
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.fill_holes(max_hole_perimeter = max_permieters)
        
        return (mesh_copy,)         

class Trellis2LaplacianSmoothingWithOpen3d:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "iterations": ("INT",{"default":10, "min":1, "max":100}),
                "method": (["Laplacian", "Taubin"],{"default":"Laplacian"}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, iterations, method):
        import open3d
        mesh_copy = copy.deepcopy(mesh)
        vertices = mesh_copy.vertices.cpu().numpy()
        faces = mesh_copy.faces.cpu().numpy().astype(np.int32)
        
        open3d_mesh = open3d.geometry.TriangleMesh()
        open3d_mesh.vertices = open3d.utility.Vector3dVector(vertices)
        open3d_mesh.triangles = open3d.utility.Vector3iVector(faces)
        
        if method == "Laplacian":
            open3d_mesh = open3d_mesh.filter_smooth_laplacian(number_of_iterations=iterations)
        elif method == "Taubin":
            open3d_mesh = open3d_mesh.filter_smooth_taubin(number_of_iterations=iterations)
            
        open3d_mesh.compute_vertex_normals()
        
        new_vertices = np.asarray(open3d_mesh.vertices)
        new_faces = np.asarray(open3d_mesh.triangles)
        
        mesh_copy.vertices = torch.from_numpy(new_vertices).float().to(mesh_copy.device)
        mesh_copy.faces = torch.from_numpy(new_faces).int().to(mesh_copy.device)
        
        return (mesh_copy,)      

class Trellis2UnWrapTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60.0,"min":0.0,"max":359.9}),
                "mesh_cluster_refine_iterations": ("INT",{"default":0}),
                "mesh_cluster_global_iterations": ("INT",{"default":1}),
                "mesh_cluster_smooth_strength": ("INT",{"default":1}),                
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, mesh_cluster_threshold_cone_half_angle_rad, mesh_cluster_refine_iterations, mesh_cluster_global_iterations, mesh_cluster_smooth_strength):
        mesh_cluster_threshold_cone_half_angle_rad = np.radians(mesh_cluster_threshold_cone_half_angle_rad)
        
        mesh_copy = trimesh.copy()
        
        vertices = torch.from_numpy(mesh_copy.vertices).float().cuda()
        faces = torch.from_numpy(mesh_copy.faces).int().cuda()
        
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)     

        out_vertices, out_faces, out_uvs = cumesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
                "refine_iterations": mesh_cluster_refine_iterations,
                "global_iterations": mesh_cluster_global_iterations,
                "smooth_strength": mesh_cluster_smooth_strength,                
            },
            return_vmaps=False,
            verbose=True,
        )
        
        del cumesh
                
        mesh_copy.vertices = out_vertices.cpu().numpy()
        mesh_copy.faces = out_faces.cpu().numpy()       
        #mesh_copy.visual.uv = out_uvs.cpu().numpy()
        mesh_copy.visual = Trimesh.visual.TextureVisuals(uv=out_uvs.cpu().numpy())
        
        return (mesh_copy,)          
        
class Trellis2MeshWithVoxelCascadeGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0x7fffffff}),
                "pipeline_type": (["1024_cascade","1536_cascade"],{"default":"1024_cascade"}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "sparse_structure_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),                
                "low_res_shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "low_res_shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "low_res_shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "low_res_shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "low_res_shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "low_res_shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "low_res_shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),                
                "high_res_shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "high_res_shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "high_res_shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "high_res_shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "high_res_shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "high_res_shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "high_res_shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),                                
                "generate_texture_slat": ("BOOLEAN", {"default":True}),                
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),         
                "texture_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),                                                               
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),      
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL","BVH", )
    RETURN_NAMES = ("mesh", "bvh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, seed, pipeline_type, 
        # sparse
        sparse_structure_steps, 
        sparse_structure_guidance_strength, 
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        sparse_structure_sampler,
        sparse_structure_resolution,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,        
        # low res shape
        low_res_shape_steps, 
        low_res_shape_guidance_strength, 
        low_res_shape_guidance_rescale,
        low_res_shape_rescale_t,
        low_res_shape_sampler,
        low_res_shape_guidance_interval_start,
        low_res_shape_guidance_interval_end,
        # high res shape
        high_res_shape_steps, 
        high_res_shape_guidance_strength, 
        high_res_shape_guidance_rescale,
        high_res_shape_rescale_t,
        high_res_shape_sampler,
        high_res_shape_guidance_interval_start,
        high_res_shape_guidance_interval_end,        
        # texture,
        generate_texture_slat,
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale,
        texture_rescale_t,        
        texture_sampler,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        # others
        max_num_tokens,
        use_tiled_decoder,
        max_views,
        fill_holes,
        hole_iterations,
        verbose,
        dino_lock,
        dino_substeps,
        hole_fill_algorithm,
        dino_foundation_cap,
        keep_only_shell
        ):
            
        reset_cuda()
        
        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images
        
        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]
        low_res_shape_guidance_interval = [low_res_shape_guidance_interval_start, low_res_shape_guidance_interval_end]
        high_res_shape_guidance_interval = [high_res_shape_guidance_interval_start, high_res_shape_guidance_interval_end]
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]
        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}        
        low_res_shape_slat_sampler_params = {"steps":low_res_shape_steps,"guidance_strength":low_res_shape_guidance_strength,"guidance_rescale":low_res_shape_guidance_rescale,"guidance_interval":low_res_shape_guidance_interval,"rescale_t":low_res_shape_rescale_t}
        high_res_shape_slat_sampler_params = {"steps":high_res_shape_steps,"guidance_strength":high_res_shape_guidance_strength,"guidance_rescale":high_res_shape_guidance_rescale,"guidance_interval":high_res_shape_guidance_interval,"rescale_t":high_res_shape_rescale_t}       
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
            
        if generate_texture_slat:
            num_steps = 5
        else:
            num_steps = 4

        pbar = ProgressBar(num_steps)
        
        mesh = pipeline.run_cascade(image=image_in, 
                                    seed=seed, 
                                    pipeline_type=pipeline_type, 
                                    sparse_structure_sampler_params = sparse_structure_sampler_params, 
                                    low_res_shape_slat_sampler_params = low_res_shape_slat_sampler_params, 
                                    high_res_shape_slat_sampler_params = high_res_shape_slat_sampler_params,
                                    tex_slat_sampler_params = tex_slat_sampler_params, 
                                    max_num_tokens = max_num_tokens, 
                                    sparse_structure_resolution = sparse_structure_resolution, 
                                    max_views = max_views, 
                                    generate_texture_slat=generate_texture_slat, 
                                    use_tiled=use_tiled_decoder, 
                                    pbar=pbar,
                                    sparse_structure_sampler = sparse_structure_sampler,
                                    low_res_shape_sampler = low_res_shape_sampler,
                                    high_res_shape_sampler = high_res_shape_sampler,
                                    tex_sampler = texture_sampler,
                                    fill_holes = fill_holes,
                                    hole_iterations = hole_iterations,
                                    verbose = verbose,
                                    dino_lock = dino_lock,
                                    dino_substeps = dino_substeps,
                                    hole_fill_algorithm = hole_fill_algorithm,
                                    dino_foundation_cap = dino_foundation_cap,
                                    keep_only_shell = keep_only_shell
                                    )[0]         
        
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()                
        
        if generate_texture_slat:
            # Build BVH for the current mesh to guide remeshing
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())           
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()
        else:
            print("Not building BVH : only used for texturing")
            bvh = None
        
        return (mesh,bvh,)      
        
class Trellis2ImageCondGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "max_views": ("INT", {"default": 1, "min": 1, "max": 999}),
            },
            "optional": {
                "naf_target_size": ("INT", {"default": 256, "min": 128, "max": 1024, "step": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE_COND", "IMAGE_COND", "TRELLIS2PIPELINE", "MOGE_CAM_CONFIG")
    RETURN_NAMES = ("cond_512", "cond_1024", "pipeline", "moge_camera_config")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, max_views, naf_target_size=256):                                   
        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images        
        
        if isinstance(image_in, (list, tuple)):
            images = list(image_in)
        else:
            images = [image_in]
            
        pipeline.load_image_cond_model()           
        
        cond_512  = pipeline.get_cond(images, 512, max_views = max_views)        
        cond_1024 = pipeline.get_cond(images, 1024, max_views = max_views)
        
        if pipeline.isPixal3D:
            pipeline.load_moge_model()
            
            if isinstance(images, list):
                image = images[0]
            else:
                image = images
                
            camera_config = pipeline.get_moge_camera_config(image)
            
            if not pipeline.keep_models_loaded:
                pipeline.unload_moge_model()            
        else:
            camera_config = None
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_image_cond_model()            

        return (cond_512, cond_1024, pipeline, camera_config)       

class Trellis2SparseGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_cond": ("IMAGE_COND",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "sparse_structure_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
            "optional":{
                "image":("IMAGE",),
                "moge_camera_config":("MOGE_CAM_CONFIG",)
            }
        }

    RETURN_TYPES = ("COORDS", "INT", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("coords", "sparse_structure_resolution", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_cond, seed, 
        # sparse
        sparse_structure_steps, 
        sparse_structure_guidance_strength, 
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        sparse_structure_sampler,
        sparse_structure_resolution,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,
        fill_holes,
        hole_iterations,
        verbose,
        dino_lock,
        dino_substeps,
        hole_fill_algorithm,
        dino_foundation_cap,
        keep_only_shell,
        image = None,
        moge_camera_config = None
        ):               
        self.seed_all(seed)
        
        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}                    

        args = pipeline._pretrained_args
        sparse_sampler_prefix = pipeline.GetSamplerName(sparse_structure_sampler)
        pipeline.sparse_structure_sampler = getattr(samplers, f"Flow{sparse_sampler_prefix}GuidanceIntervalSampler")(**args['sparse_structure_sampler']['args'])

        if pipeline.isPixal3D:
            if image is not None:
                images = tensor_batch_to_pil_list(image, max_views=16)                
                images = list(images)
            else:
                raise Exception('Image is required for Pixal3D')
                
            if moge_camera_config is not None:
                camera_angle_x = moge_camera_config['camera_angle_x']
                distance = moge_camera_config['distance']
                mesh_scale = moge_camera_config['mesh_scale']
            else:
                raise Exception('MoGe Camera Config is required for Pixal3D')

            image_cond_model = pipeline.load_pixal3d_image_cond_ss()            
        
            image_cond = pipeline.get_proj_cond_ss(
                image=images,
                camera_angle_x=camera_angle_x,
                distance=distance,
                mesh_scale=mesh_scale,
                image_cond_model=image_cond_model
            )
            
            if not pipeline.keep_models_loaded:
                pipeline.unload_pixal3d_image_cond_ss()
                
        pipeline.load_sparse_structure_model()
        
        coords = pipeline.sample_sparse_structure(
            image_cond, sparse_structure_resolution,
            1, sparse_structure_sampler_params,
            fill_holes=fill_holes,
            hole_iterations=hole_iterations,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            hole_fill_algorithm=hole_fill_algorithm,
            dino_foundation_cap=dino_foundation_cap,
            keep_only_shell=keep_only_shell
        )
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_sparse_structure_model()            

        return (coords, sparse_structure_resolution, pipeline,)
        
    def seed_all(self, seed: int = 0):
        import random
        """
        Set random seeds of all components.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)         
        
class Trellis2ShapeGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_cond": ("IMAGE_COND",),
                "coords": ("COORDS",),
                "resolution": ([512,1024],{"default":1024}),                
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),                
            },
            "optional":
            {
                "image": ("IMAGE",),
                "moge_camera_config": ("MOGE_CAM_CONFIG",),
            }    
        }

    RETURN_TYPES = ("SHAPE_SLAT", "INT", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("shape_slat", "resolution", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_cond, coords, resolution,      
        # shape
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,
        shape_sampler,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        image = None,
        moge_camera_config = None
        ):
            
        shape_guidance_interval = [shape_guidance_interval_start, shape_guidance_interval_end]        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}            
        
        args = pipeline._pretrained_args
        shape_sampler_prefix = pipeline.GetSamplerName(shape_sampler)
        pipeline.shape_slat_sampler = getattr(samplers, f"Flow{shape_sampler_prefix}GuidanceIntervalSampler")(**args['shape_slat_sampler']['args'])                    
        
        if resolution == 512:
            pipeline.unload_shape_slat_flow_model_1024()
            
            if pipeline.isPixal3D:
                images = tensor_batch_to_pil_list(image, max_views=16)
                image_in = images[0] if len(images) == 1 else images        
                
                if isinstance(image_in, (list, tuple)):
                    images = list(image_in)
                else:
                    images = [image_in]
                    
                camera_angle_x = moge_camera_config['camera_angle_x']
                distance = moge_camera_config['distance']
                mesh_scale = moge_camera_config['mesh_scale']
                
                image_cond_model = pipeline.load_pixal3d_image_cond_shape_512()
                    
                image_cond = pipeline.get_proj_cond_shape(
                    image_cond_model, images, coords,
                    camera_angle_x=camera_angle_x,
                    distance=distance,
                    mesh_scale=mesh_scale,
                )
                
                if not pipeline.keep_models_loaded:
                    pipeline.unload_pixal3d_image_cond_shape_512()
            
            pipeline.load_shape_slat_flow_model_512()
            
            shape_slat = pipeline.sample_shape_slat(
                image_cond, pipeline.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params,
                verbose = verbose,
                dino_lock = dino_lock,
                dino_substeps = dino_substeps,
                dino_foundation_cap = dino_foundation_cap
            )
            
            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_flow_model_512()
                
        elif resolution == 1024:
            pipeline.unload_shape_slat_flow_model_512()            
            
            if pipeline.isPixal3D:
                images = tensor_batch_to_pil_list(image, max_views=16)
                image_in = images[0] if len(images) == 1 else images        
                
                if isinstance(image_in, (list, tuple)):
                    images = list(image_in)
                else:
                    images = [image_in]
                    
                camera_angle_x = moge_camera_config['camera_angle_x']
                distance = moge_camera_config['distance']
                mesh_scale = moge_camera_config['mesh_scale']
                
                image_cond_model = pipeline.load_pixal3d_image_cond_shape_1024()               
                    
                image_cond = pipeline.get_proj_cond_shape(
                    image_cond_model, images, coords,
                    camera_angle_x=camera_angle_x,
                    distance=distance,
                    mesh_scale=mesh_scale,
                )
                
                if not pipeline.keep_models_loaded:
                    pipeline.unload_pixal3d_image_cond_shape_1024()
            
            pipeline.load_shape_slat_flow_model_1024()
            
            shape_slat = pipeline.sample_shape_slat(
                image_cond, pipeline.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params,
                verbose = verbose,
                dino_lock = dino_lock,
                dino_substeps = dino_substeps,
                dino_foundation_cap = dino_foundation_cap
            )
            
            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_flow_model_1024()                
        
        return (shape_slat, resolution, pipeline,)      

class Trellis2ShapeCascadeGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_cond": ("IMAGE_COND",),
                "shape_slat": ("SHAPE_SLAT",),
                "from_resolution": ("INT",),
                "to_resolution": ([1024,1536],{"default":1024}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
            },
            "optional":
            {
                "image": ("IMAGE",),
                "moge_camera_config": ("MOGE_CAM_CONFIG",),
            }              
        }

    RETURN_TYPES = ("SHAPE_SLAT","INT","TRELLIS2PIPELINE","INT",)
    RETURN_NAMES = ("shape_slat","resolution","pipeline","num_tokens")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_cond, shape_slat, from_resolution, to_resolution, sparse_structure_resolution, max_num_tokens,      
        # shape
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,
        shape_sampler,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        image = None,
        moge_camera_config = None
        ):
            
        shape_guidance_interval = [shape_guidance_interval_start, shape_guidance_interval_end]        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}                    
        
        args = pipeline._pretrained_args
        shape_sampler_prefix = pipeline.GetSamplerName(shape_sampler)
        pipeline.shape_slat_sampler = getattr(samplers, f"Flow{shape_sampler_prefix}GuidanceIntervalSampler")(**args['shape_slat_sampler']['args'])
        slat, hr_resolution, num_tokens = self.sample(pipeline, shape_slat, from_resolution, to_resolution, sparse_structure_resolution, max_num_tokens, image_cond, shape_slat_sampler_params, verbose, dino_lock, dino_substeps, dino_foundation_cap, image, moge_camera_config)
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_shape_slat_flow_model_1024()              
        
        return (slat, hr_resolution, pipeline, num_tokens,)         
        
    def sample(self, pipeline, slat, lr_resolution, resolution, sparse_structure_resolution, max_num_tokens, cond, sampler_params, verbose, dino_lock, dino_substeps, dino_foundation_cap, image, moge_camera_config):
        # Upsample       
        pipeline.load_shape_slat_decoder()
        if pipeline.low_vram:
            pipeline.models['shape_slat_decoder'].to(pipeline.device)
            pipeline.models['shape_slat_decoder'].low_vram = True
        hr_coords = pipeline.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if pipeline.low_vram:
            pipeline.models['shape_slat_decoder'].cpu()
            pipeline.models['shape_slat_decoder'].low_vram = False
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_shape_slat_decoder()
        
        hr_resolution = resolution
        ratio = (sparse_structure_resolution / 32)
        
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / (lr_resolution * ratio) * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                print(f"Num Tokens: {num_tokens}")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 512
                break
                
        if pipeline.isPixal3D:
            images = tensor_batch_to_pil_list(image, max_views=16)
            image_in = images[0] if len(images) == 1 else images        
            
            if isinstance(image_in, (list, tuple)):
                images = list(image_in)
            else:
                images = [image_in]
                
            camera_angle_x = moge_camera_config['camera_angle_x']
            distance = moge_camera_config['distance']
            mesh_scale = moge_camera_config['mesh_scale']
            
            image_cond_model = pipeline.load_pixal3d_image_cond_shape_1024()
                
            actual_grid_res = hr_resolution // 16
                
            cond = pipeline.get_proj_cond_shape(
                image_cond_model, images, coords,
                camera_angle_x=camera_angle_x,
                distance=distance,
                mesh_scale=mesh_scale,
                grid_resolution_override=actual_grid_res,
            )
            
            if not pipeline.keep_models_loaded:
                pipeline.unload_pixal3d_image_cond_shape_1024()
        
        pipeline.load_shape_slat_flow_model_1024()
        flow_model = pipeline.models['shape_slat_flow_model_1024']
        
        if pipeline.low_vram:
            cond = pipeline._cond_to(cond, pipeline.device)                
        
        coords_dev = coords.to(pipeline.device)                                           
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=pipeline.device),
            coords=coords_dev,
        )
        sampler_params = {**pipeline.shape_slat_sampler_params, **sampler_params}
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
        slat = pipeline.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        if pipeline.low_vram:
            flow_model.cpu()
            pipeline._cleanup_cuda()                                

        std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if pipeline.low_vram:
            cond = pipeline._cond_cpu(cond)
            pipeline._cleanup_cuda()            

        return slat, hr_resolution, num_tokens 

class Trellis2TexSlatGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_cond": ("IMAGE_COND",),
                "shape_slat": ("SHAPE_SLAT",),
                "resolution": ([512,1024],{"default":1024}),                
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),         
                "texture_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),                                                               
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
            },
            "optional":
            {
                "image": ("IMAGE",),
                "moge_camera_config": ("MOGE_CAM_CONFIG",),
                "from_resolution": ("INT",),
            }            
        }

    RETURN_TYPES = ("TEXTURE_SLAT", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("texture_slat", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_cond, shape_slat, resolution,      
        # shape
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale,
        texture_rescale_t,
        texture_sampler,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        image = None,
        moge_camera_config = None,
        from_resolution = None
        ):

        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
        
        if resolution == 512:
            if pipeline.isPixal3D:
                raise Exception('Pixal3D only works with 1024 resolution')
                
            pipeline.unload_tex_slat_flow_model_1024()
            pipeline.load_tex_slat_flow_model_512()           
            
            tex_slat = pipeline.sample_tex_slat_advanced(
                image_cond, pipeline.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params, texture_sampler,
                verbose = verbose,
                dino_lock = dino_lock,
                dino_substeps = dino_substeps,
                dino_foundation_cap = dino_foundation_cap
            )
            if not pipeline.keep_models_loaded:
                pipeline.unload_tex_slat_flow_model_512()
                
        elif resolution == 1024:
            if not pipeline.isPixal3D:
                pipeline.unload_tex_slat_flow_model_512()            
            
            if pipeline.isPixal3D:
                images = tensor_batch_to_pil_list(image, max_views=16)
                image_in = images[0] if len(images) == 1 else images        
                
                if isinstance(image_in, (list, tuple)):
                    images = list(image_in)
                else:
                    images = [image_in]
                    
                camera_angle_x = moge_camera_config['camera_angle_x']
                distance = moge_camera_config['distance']
                mesh_scale = moge_camera_config['mesh_scale']
                
                image_cond_model = pipeline.load_pixal3d_image_cond_tex_1024()
                
                tex_grid_res = from_resolution // 16
                
                image_cond = pipeline.get_proj_cond_shape(
                    image_cond_model, images, shape_slat.coords,
                    camera_angle_x=camera_angle_x,
                    distance=distance,
                    mesh_scale=mesh_scale,
                    grid_resolution_override=tex_grid_res,
                )
                
                if not pipeline.keep_models_loaded:
                    pipeline.unload_pixal3d_image_cond_tex_1024()
            
            pipeline.load_tex_slat_flow_model_1024()
            
            tex_slat = pipeline.sample_tex_slat_advanced(
                image_cond, pipeline.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params, texture_sampler,
                verbose = verbose,
                dino_lock = dino_lock,
                dino_substeps = dino_substeps,
                dino_foundation_cap = dino_foundation_cap
            )
            
            if not pipeline.keep_models_loaded:
                pipeline.unload_tex_slat_flow_model_1024()                
        
        return (tex_slat, pipeline,)      
        
class Trellis2DecodeLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "shape_slat": ("SHAPE_SLAT",),
                "resolution": ("INT",),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
            },
            "optional": {
                "texture_slat": ("TEXTURE_SLAT",),
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL", "BVH", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("mesh", "bvh", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, shape_slat, resolution, use_tiled_decoder, texture_slat = None):
        mesh = pipeline.decode_latent(shape_slat, texture_slat, resolution, use_tiled=use_tiled_decoder)[0]
        
        if texture_slat == None:
            print("Not building BVH : only used for texturing")
            bvh = None            
        else:
            # Build BVH for the current mesh to guide remeshing
            vertices = mesh.vertices.cuda()
            faces = mesh.faces.cuda()   
            
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())           
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()            
        
        
        return (mesh, bvh, pipeline,)    

class Trellis2SimplifyMeshAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "target_face_num": ("INT",{"default":1000000,"min":1,"max":30000000}),
                "thresh":("FLOAT",{"default":1e-8,"min":1e-12,"max":1e-2,"step":0.000000000001}),
                "lambda_edge_length": ("FLOAT",{"default":0.01,"min":0.00,"max":1.00,"step":0.01}),
                "lambda_skinny": ("FLOAT",{"default":0.001,"min":0.000,"max":0.100,"step":0.001}),
                "lambda_curvature": ("FLOAT",{"default":0.050,"min":0.000,"max":0.500,"step":0.001}),
                "lambda_boundary": ("FLOAT",{"default":0.050,"min":0.000,"max":0.500,"step":0.001}),
                "lambda_area": ("FLOAT",{"default":0.010,"min":0.000,"max":0.100,"step":0.001}),
                "qem_regularization": ("FLOAT",{"default":1e-8,"min":1e-10,"max":1e-5,"step":0.0000000001}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, target_face_num, thresh, lambda_edge_length, lambda_skinny, lambda_curvature, lambda_boundary, lambda_area, qem_regularization):        
        mesh_copy = copy.deepcopy(mesh)

        options = {
            'method': 'advanced',
            'thresh': thresh,
            'lambda_edge_length': lambda_edge_length,
            'lambda_skinny': lambda_skinny,
            'lambda_curvature': lambda_curvature,
            'lambda_boundary': lambda_boundary,            
            'lambda_area': lambda_area,     
            'qem_regularization': qem_regularization,     
        }         

        mesh_copy.simplify_with_cumesh(target = target_face_num, options = options)
        
        return (mesh_copy,)  

class Trellis2SimplifyTrimeshAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_face_num": ("INT",{"default":1000000,"min":1,"max":30000000}),
                "thresh":("FLOAT",{"default":1e-8,"min":1e-12,"max":1e-2,"step":0.000000000001}),
                "lambda_edge_length": ("FLOAT",{"default":0.01,"min":0.00,"max":1.00,"step":0.01}),
                "lambda_skinny": ("FLOAT",{"default":0.001,"min":0.000,"max":0.100,"step":0.001}),
                "lambda_curvature": ("FLOAT",{"default":0.050,"min":0.000,"max":0.500,"step":0.001}),
                "lambda_boundary": ("FLOAT",{"default":0.050,"min":0.000,"max":0.500,"step":0.001}),
                "lambda_area": ("FLOAT",{"default":0.010,"min":0.000,"max":0.100,"step":0.001}),
                "qem_regularization": ("FLOAT",{"default":1e-8,"min":1e-10,"max":1e-5,"step":0.0000000001}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, target_face_num, thresh, lambda_edge_length, lambda_skinny, lambda_curvature, lambda_boundary, lambda_area, qem_regularization):
        mesh_copy = copy.deepcopy(trimesh)
        
        cumesh = CuMesh.CuMesh()
        cumesh.init(torch.from_numpy(mesh_copy.vertices).float().cuda(), torch.from_numpy(mesh_copy.faces).int().cuda())
        
        options = {
            'method': 'advanced',
            'thresh': thresh,
            'lambda_edge_length': lambda_edge_length,
            'lambda_skinny': lambda_skinny,
            'lambda_curvature': lambda_curvature,
            'lambda_boundary': lambda_boundary,            
            'lambda_area': lambda_area,     
            'qem_regularization': qem_regularization,     
        }        

        cumesh.simplify(target_face_num, verbose=True, options = options)
        
        new_vertices, new_faces = cumesh.read()
        mesh_copy.vertices = new_vertices.cpu().numpy()
        mesh_copy.faces = new_faces.cpu().numpy()
            
        del cumesh        
        
        return (mesh_copy,)

class Trellis2MultiViewTexturing:
    """
    Apply texture to mesh by projecting multiple view images.
    
    Uses angle-weighted blending: each surface receives texture from all views
    that can "see" it, weighted by how directly the surface faces each camera.
    
    Camera angles (Y-up coordinate system):
    - Azimuth: rotation around Y axis
      - 0° = front (looking in -Z direction)
      - 90° = left (looking in -X direction)  
      - 180° = back (looking in +Z direction)
      - 270° = right (looking in +X direction)
    - Elevation: rotation around X axis
      - 0° = horizontal
      - 90° = top (looking in -Y direction, from above)
      - -90° = bottom (looking in +Y direction, from below)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "texture_size": ("INT", {"default": 4096, "min": 512, "max": 8192}),
                "blend_texture": ("BOOLEAN", {"default":True}),
                "blend_exponent": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 99.0, "step": 0.1}),
                "ortho_scale": ("FLOAT", {"default": 1.1, "min": 0.05, "max": 10.0, "step": 0.01}),
                "norm_size": ("FLOAT",{"default":1.15, "min":0.0, "max":9.99, "step":0.01}),
                "fill_holes": ("BOOLEAN",{"default":True}),
                "max_hole_size": ("INT",{"default":20,"min":0,"max":99999,"step":1}),
                "use_metallic": ("BOOLEAN",{"default":True}),
                "depth_eps": ("FLOAT",{"default":0.0100,"min":0.0001,"max":1.0000,"step":0.0001}),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60,"min":1,"max":179,"step":1}),
                "add_alpha_channel": ("BOOLEAN",{"default":False}),
            },
            "optional": {
                # Standard views
                "front_image": ("IMAGE",),   # az=0, el=0
                "back_image": ("IMAGE",),    # az=180, el=0
                "left_image": ("IMAGE",),    # az=90, el=0
                "right_image": ("IMAGE",),   # az=270, el=0
                "top_image": ("IMAGE",),     # az=0, el=90
                "bottom_image": ("IMAGE",),  # az=0, el=-90
                "front_weight": ("FLOAT",{"default":1.000,"min":0.001,"max":1.000,"step":0.001}),
                "back_weight": ("FLOAT",{"default":1.000,"min":0.001,"max":1.000,"step":0.001}),
                "left_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "right_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "top_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "bottom_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                # Custom views
                "custom_images": ("IMAGE",),
                "custom_azimuths": ("STRING", {"default": ""}),
                "custom_elevations": ("STRING", {"default": ""}),
                "custom_weights": ("STRING", {"default": ""}),
                "camera_config": ("HY3DCAMERA",),
            }
        }
    
    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("trimesh", "base_color", "metallic_roughness",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True
    
    def process(
        self,
        trimesh,
        texture_size,
        blend_texture,
        blend_exponent,
        ortho_scale,
        norm_size,
        fill_holes,
        max_hole_size,
        use_metallic,
        depth_eps,
        mesh_cluster_threshold_cone_half_angle_rad,
        add_alpha_channel,
        baseColorTexture = None,
        front_image=None,
        back_image=None,
        left_image=None,
        right_image=None,
        top_image=None,
        bottom_image=None,
        front_weight=None,
        back_weight=None,
        left_weight=None,
        right_weight=None,
        top_weight=None,
        bottom_weight=None,
        custom_images=None,
        custom_azimuths="",
        custom_elevations="",
        custom_weights="",
        camera_config = None
    ):
        from .projection.texture_projection_multiview import texture_mesh_with_multiview
        
        reset_cuda()
        
        # Collect views
        images = []
        azimuths = []
        elevations = []
        weights = []
        
        # Standard views with their camera angles
        standard_views = [
            (front_image, 0, 0, "front", front_weight),
            (back_image, 180, 0, "back", back_weight),
            (left_image, 90, 0, "left", left_weight),
            (right_image, 270, 0, "right", right_weight),
            (top_image, 0, 90, "top", top_weight),
            (bottom_image, 0, -90, "bottom", bottom_weight),
        ]
        
        for img, az, el, name, w in standard_views:
            if img is not None:
                images.append(self._tensor_to_pil(img))
                azimuths.append(az)
                elevations.append(el)
                weights.append(w)
                print(f"[MultiView] Added {name} view (az={az}, el={el}, w={w})")
        
        # Custom views
        if custom_images is not None:
            custom_az_list = self._parse_angles(custom_azimuths)
            custom_el_list = self._parse_angles(custom_elevations)
            custom_w_list = self._parse_angles(custom_weights)
            
            if custom_az_list and custom_el_list:
                num_custom = min(len(custom_az_list), len(custom_el_list), int(custom_images.shape[0]), len(custom_w_list))
                for i in range(num_custom):
                    images.append(self._tensor_to_pil(custom_images[i:i+1]))
                    azimuths.append(custom_az_list[i])
                    elevations.append(custom_el_list[i])
                    weights.append(custom_w_list[i])
                    print(f"[MultiView] Added custom view {i+1} (az={custom_az_list[i]}, el={custom_el_list[i]})")
            elif camera_config:
                selected_camera_azims = camera_config["selected_camera_azims"]
                selected_camera_elevs = camera_config["selected_camera_elevs"]
                selected_view_weights = camera_config["selected_view_weights"]
                #ortho_scale = camera_config["ortho_scale"]             

                num_custom = min(len(selected_camera_azims), len(selected_camera_elevs), int(custom_images.shape[0]))
                for i in range(num_custom):
                    images.append(self._tensor_to_pil(custom_images[i:i+1]))
                    azimuths.append(selected_camera_azims[i])
                    elevations.append(selected_camera_elevs[i])
                    weights.append(selected_view_weights[i])
                    print(f"[MultiView] Added custom view {i+1} (az={selected_camera_azims[i]}, el={selected_camera_elevs[i]}, w={selected_view_weights[i]})")                
        
        if len(images) == 0:
            raise ValueError("No input images provided! Please connect at least one image.")
        
        print(f"[MultiView] Total views: {len(images)}")
        print(f"[MultiView] Azimuths: {azimuths}")
        print(f"[MultiView] Elevations: {elevations}")

        trimesh_obj, base_color, mr = texture_mesh_with_multiview(
            trimesh,
            images,
            azimuths,
            elevations,
            weights,
            texture_size=texture_size,
            mesh_cluster_threshold_cone_half_angle_rad=mesh_cluster_threshold_cone_half_angle_rad,
            blend_exponent=blend_exponent,
            ortho_scale=ortho_scale,
            blend_texture=blend_texture,
            fill_holes=fill_holes,
            norm_size=norm_size,
            max_hole_size=max_hole_size,
            use_metallic=use_metallic,
            depth_eps=depth_eps,
            add_alpha_channel=add_alpha_channel
        )
        
        return (trimesh_obj, pil2tensor(base_color), pil2tensor(mr))
    
    def _tensor_to_pil(self, tensor):
        """Convert ComfyUI IMAGE tensor to PIL."""
        if len(tensor.shape) == 4:
            arr = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def _parse_angles(self, angle_string):
        """Parse comma-separated angles into list of floats."""
        if not angle_string or angle_string.strip() == "":
            return []
        try:
            return [float(x.strip()) for x in angle_string.split(",") if x.strip()]
        except ValueError:
            print(f"[MultiView] Warning: Could not parse angles: {angle_string}")
            return []
            
class Trellis2ProjectHighPolyToLowPoly:
    """
    Apply texture to mesh by projecting multiple view images.
    
    Uses angle-weighted blending: each surface receives texture from all views
    that can "see" it, weighted by how directly the surface faces each camera.
    
    Camera angles (Y-up coordinate system):
    - Azimuth: rotation around Y axis
      - 0° = front (looking in -Z direction)
      - 90° = left (looking in -X direction)  
      - 180° = back (looking in +Z direction)
      - 270° = right (looking in +X direction)
    - Elevation: rotation around X axis
      - 0° = horizontal
      - 90° = top (looking in -Y direction, from above)
      - -90° = bottom (looking in +Y direction, from below)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "high_poly_trimesh": ("TRIMESH",),
                "low_poly_trimesh": ("TRIMESH",),
                "texture_size": ("INT", {"default": 4096, "min": 512, "max": 8192}),
                "blend_texture": ("BOOLEAN", {"default":True}),
                "blend_exponent": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 99.0, "step": 0.5}),
                "ortho_scale": ("FLOAT", {"default": 1.1, "min": 0.05, "max": 10.0, "step": 0.01}),
                "norm_size": ("FLOAT",{"default":1.15, "min":0.0, "max":9.99, "step":0.01}),
                "fill_holes": ("BOOLEAN",{"default":True}),
                "max_hole_size": ("INT",{"default":20,"min":0,"max":99999,"step":1}),
                "use_metallic": ("BOOLEAN",{"default":False}),
                "depth_eps": ("FLOAT",{"default":0.0100,"min":0.0001,"max":1.0000,"step":0.0001}),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60,"min":1,"max":179,"step":1}),
                "add_alpha_channel": ("BOOLEAN",{"default":False}),                
            },
            "optional": {
                # Standard views
                "front_image": ("IMAGE",),   # az=0, el=0
                "back_image": ("IMAGE",),    # az=180, el=0
                "left_image": ("IMAGE",),    # az=90, el=0
                "right_image": ("IMAGE",),   # az=270, el=0
                "top_image": ("IMAGE",),     # az=0, el=90
                "bottom_image": ("IMAGE",),  # az=0, el=-90
                "front_weight": ("FLOAT",{"default":1.000,"min":0.001,"max":1.000,"step":0.001}),
                "back_weight": ("FLOAT",{"default":1.000,"min":0.001,"max":1.000,"step":0.001}),
                "left_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "right_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "top_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "bottom_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                # Custom views
                "custom_images": ("IMAGE",),
                "custom_azimuths": ("STRING", {"default": ""}),
                "custom_elevations": ("STRING", {"default": ""}),
                "custom_weights": ("STRING", {"default": ""}),
                "camera_config": ("HY3DCAMERA",),
            }
        }
    
    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("trimesh", "base_color", "metallic_roughness",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True
    
    def process(
        self,
        high_poly_trimesh,
        low_poly_trimesh,
        texture_size,
        blend_texture,
        blend_exponent,
        ortho_scale,
        norm_size,
        fill_holes,
        max_hole_size,
        use_metallic,
        depth_eps,
        mesh_cluster_threshold_cone_half_angle_rad,
        add_alpha_channel,
        baseColorTexture = None,
        front_image=None,
        back_image=None,
        left_image=None,
        right_image=None,
        top_image=None,
        bottom_image=None,
        front_weight=None,
        back_weight=None,
        left_weight=None,
        right_weight=None,
        top_weight=None,
        bottom_weight=None,
        custom_images=None,
        custom_azimuths="",
        custom_elevations="",
        custom_weights="",
        camera_config = None,
    ):
        from .projection.texture_projection_multiview import texture_mesh_with_multiview
        
        reset_cuda()
        
        # Collect views
        images = []
        azimuths = []
        elevations = []
        weights = []
        
        # Standard views with their camera angles
        standard_views = [
            (front_image, 0, 0, "front", front_weight),
            (back_image, 180, 0, "back", back_weight),
            (left_image, 90, 0, "left", left_weight),
            (right_image, 270, 0, "right", right_weight),
            (top_image, 0, 90, "top", top_weight),
            (bottom_image, 0, -90, "bottom", bottom_weight),
        ]
        
        for img, az, el, name, w in standard_views:
            if img is not None:
                images.append(self._tensor_to_pil(img))
                azimuths.append(az)
                elevations.append(el)
                weights.append(w)
                print(f"[MultiView] Added {name} view (az={az}, el={el}, w={w})")        
                    
        # Custom views
        if custom_images is not None:
            custom_az_list = self._parse_angles(custom_azimuths)
            custom_el_list = self._parse_angles(custom_elevations)
            custom_w_list = self._parse_angles(custom_weights)
            
            if custom_az_list and custom_el_list:
                num_custom = min(len(custom_az_list), len(custom_el_list), int(custom_images.shape[0]), len(custom_w_list))
                for i in range(num_custom):
                    images.append(self._tensor_to_pil(custom_images[i:i+1]))
                    azimuths.append(custom_az_list[i])
                    elevations.append(custom_el_list[i])
                    weights.append(custom_w_list[i])
                    print(f"[MultiView] Added custom view {i+1} (az={custom_az_list[i]}, el={custom_el_list[i]})")
            elif camera_config:
                selected_camera_azims = camera_config["selected_camera_azims"]
                selected_camera_elevs = camera_config["selected_camera_elevs"]
                selected_view_weights = camera_config["selected_view_weights"]
                #ortho_scale = camera_config["ortho_scale"]             

                num_custom = min(len(selected_camera_azims), len(selected_camera_elevs), int(custom_images.shape[0]))
                for i in range(num_custom):
                    images.append(self._tensor_to_pil(custom_images[i:i+1]))
                    azimuths.append(selected_camera_azims[i])
                    elevations.append(selected_camera_elevs[i])
                    weights.append(selected_view_weights[i])
                    print(f"[MultiView] Added custom view {i+1} (az={selected_camera_azims[i]}, el={selected_camera_elevs[i]}, w={selected_view_weights[i]})")                
        
        if len(images) == 0:
            raise ValueError("No input images provided! Please connect at least one image.")
        
        print(f"[MultiView] Total views: {len(images)}")
        print(f"[MultiView] Azimuths: {azimuths}")
        print(f"[MultiView] Elevations: {elevations}")

        trimesh_obj, base_color, mr = texture_mesh_with_multiview(
            high_poly_trimesh,
            images,
            azimuths,
            elevations,
            weights,
            texture_size=texture_size,
            mesh_cluster_threshold_cone_half_angle_rad=mesh_cluster_threshold_cone_half_angle_rad,
            blend_exponent=blend_exponent,
            ortho_scale=ortho_scale,
            blend_texture=blend_texture,
            fill_holes=fill_holes,
            norm_size=norm_size,
            max_hole_size=max_hole_size,
            use_metallic=use_metallic,
            depth_eps=depth_eps,
            low_poly_mesh=low_poly_trimesh,
            add_alpha_channel=add_alpha_channel
        )
        
        return (trimesh_obj, pil2tensor(base_color), pil2tensor(mr))
    
    def _tensor_to_pil(self, tensor):
        """Convert ComfyUI IMAGE tensor to PIL."""
        if len(tensor.shape) == 4:
            arr = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def _parse_angles(self, angle_string):
        """Parse comma-separated angles into list of floats."""
        if not angle_string or angle_string.strip() == "":
            return []
        try:
            return [float(x.strip()) for x in angle_string.split(",") if x.strip()]
        except ValueError:
            print(f"[MultiView] Warning: Could not parse angles: {angle_string}")
            return [] 

class Trellis2RenderMultiView:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "render_size": ("INT", {"default": 1024, "min": 512, "max": 8192}),
                "ortho_scale": ("FLOAT", {"default": 1.2, "min": 0.05, "max": 10.0, "step": 0.01}),
                "blender_exec_path": ("STRING",),
                "azimuths": ("STRING",{"default":"0,90,180,270,0,0"}),
                "elevations": ("STRING",{"default":"0,0,0,0,90,-90"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE","FLOAT", "STRING", "STRING",)
    RETURN_NAMES = ("images","ortho_scale", "azimuths", "elevations",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True
    
    def process(
        self,
        trimesh,
        render_size,
        ortho_scale,
        blender_exec_path,
        azimuths,
        elevations
    ):
        reset_cuda()
        
        if not hasattr(trimesh.visual, 'material'):
            raise Exception("Trimesh does not have a material")
                    
        custom_az_list = self._parse_angles(azimuths)
        custom_el_list = self._parse_angles(elevations)                    
                    
        if custom_az_list and custom_el_list:
            if len(custom_az_list) != len(custom_el_list):
                raise Exception("azimuths and elevations must have the same amount of values")
                
            textured_maps = self.render_textured_multiview(
                custom_el_list, custom_az_list, ortho_scale, render_size, blender_exec_path, trimesh)
            custom_images = torch.stack(textured_maps, dim=0)

            return (custom_images, ortho_scale, azimuths, elevations,)
        else:
            raise Exception("azimuths and elevations are required")
    
    def _parse_angles(self, angle_string):
        """Parse comma-separated angles into list of floats."""
        if not angle_string or angle_string.strip() == "":
            return []
        try:
            return [float(x.strip()) for x in angle_string.split(",") if x.strip()]
        except ValueError:
            print(f"[MultiView] Warning: Could not parse angles: {angle_string}")
            return []         

    def render_textured_multiview(self, camera_elevs, camera_azims, ortho_scale, resolution, blender_exec_path, mesh):
        from .projection.camera_utils import get_orthographic_projection_matrix
        
        proj = get_orthographic_projection_matrix(
            left=-ortho_scale * 0.5, right=ortho_scale * 0.5,
            bottom=-ortho_scale * 0.5, top=ortho_scale * 0.5,
            near=0.1, far=100
        )        
        textured_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            textured_map = self.render(
                elev, azim, filter_mode='linear', return_type='th', scale=ortho_scale, resolution=resolution, blender_exec_path=blender_exec_path,proj=proj,mesh=mesh)
            textured_maps.append(textured_map)
            
        return textured_maps

    def render(
        self,
        elev,
        azim,
        camera_distance=None,
        center=None,
        resolution=None,
        tex=None,
        keep_alpha=False,
        bgcolor=None,
        filter_mode=None,
        return_type='th',
        scale=1.0,
        blender_exec_path=None,
        proj=None,
        mesh=None,
    ):
        from .projection.camera_utils import get_mv_matrix
        
        r_mv = get_mv_matrix(
            elev=elev,
            azim=azim,
            camera_distance=1.1,
            center=center)
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        if tex is not None:
            if isinstance(tex, Image.Image):
                tex = torch.tensor(np.array(tex) / 255.0)
            elif isinstance(tex, np.ndarray):
                tex = torch.tensor(tex)
            if tex.dim() == 2:
                tex = tex.unsqueeze(-1)
            tex = tex.float().to(self.device)
        # image = self._render(r_mvp, self.vtx_pos, self.pos_idx, self.vtx_uv, self.uv_idx,
                             # self.tex if tex is None else tex,
                             # self.default_resolution if resolution is None else resolution,
                             # self.max_mip_level, True, filter_mode if filter_mode else self.filter_mode,
                             # elev=elev, azim=azim, camera_distance=camera_distance,scale=scale,blender_exec_path=blender_exec_path)
        image = self.raster_texture(tex, mesh.visual.uv, elev=elev, azim=azim, camera_distance=camera_distance, resolution=resolution, scale=scale, blender_exec_path=blender_exec_path,mesh=mesh)
        mask = (image[..., [-1]] == 1).float()
        if bgcolor is None:
            bgcolor = [0 for _ in range(image.shape[-1] - 1)]
        image = image * mask + (1 - mask) * \
                torch.tensor(bgcolor + [0])
        if keep_alpha == False:
            image = image[..., :-1]
        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.squeeze(-1).cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        return image      

    def raster_texture(self, tex, uv, uv_da=None, mip_level_bias=None, mip=None, filter_mode='auto',
                       boundary_mode='wrap', max_mip_level=None, elev=None, azim=None, camera_distance=None, resolution=None, scale=1.0,
                       blender_exec_path=None,mesh=None):
        import tempfile
        import subprocess
        
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp_mesh:
            mesh_path = tmp_mesh.name
            tmp_mesh.close()
        
        mesh.export(mesh_path)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
            output_path = tmp_out.name
            tmp_out.close()
            
        blender_script = os.path.join(os.path.dirname(__file__), 'projection', 'blender_render.py')
        
        res = resolution[0] if isinstance(resolution, (list, tuple)) else resolution
        
        cmd = [
            blender_exec_path, '-b', '-P', blender_script, '--',
            '--mesh', mesh_path,
            '--output', output_path,
            '--elev', str(elev),
            '--azim', str(azim),
            '--scale', str(scale),
            '--resolution', str(res)
        ]
        
        subprocess.run(cmd, check=True)
        
        image = Image.open(output_path)
        image = torch.tensor(np.array(image) / 255.0).float()
        
        if os.path.exists(mesh_path):
            os.remove(mesh_path)
        if os.path.exists(output_path):
            os.remove(output_path)
            
        return image          
            
class Trellis2CudaReset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
            },
        }

    RETURN_TYPES = (any, )
    RETURN_NAMES = ("output_1", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1):
        reset_cuda()
        return (input_1,)          
        
class Trellis2SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "compress_level": ("INT",{"default":1,"min":1,"max":9,"step":1}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("images_path",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    ESSENTIALS_CATEGORY = "Basics"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."
    SEARCH_ALIASES = ["save", "save image", "export image", "output image", "write image", "download"]

    def save_images(self, images, filename_prefix="ComfyUI", compress_level=4, prompt=None, extra_pnginfo=None):
        from comfy.cli_args import args
        from PIL.PngImagePlugin import PngInfo
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        file_list = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            full_path = os.path.join(full_output_folder, file)
            file_list.append(full_path)
            img.save(full_path, pnginfo=metadata, compress_level=compress_level)
            counter += 1

        if len(file_list)==1:
            file_list = file_list[0]
            
        return (file_list,)
        
class Trellis2VoxelToMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "target_height_mm": ("FLOAT",{"default":0.0,"min":0.0,"max":500.0,"step":0.1}),
                "sigma": ("FLOAT",{"default":0.0,"min":0.0,"max":9.9,"step":0.1}),
                "coarse_downsample": ("FLOAT",{"default":1.00,"min":1.00,"max":9.00,"step":0.01}),
                "taubin_iterations": ("INT",{"default":0,"min":0,"max":999,"step":1}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, target_height_mm, sigma, coarse_downsample, taubin_iterations):
        mesh_copy = copy.deepcopy(mesh)
        
        if mesh_copy.coords is None:
            raise Exception("Voxel to Mesh requires Texture Slat")
        else:
            from .blackwell_fix import voxel_to_mesh
            
            trimesh = voxel_to_mesh(mesh_output = mesh_copy, 
                                    target_height_mm = target_height_mm,
                                    sigma = sigma,
                                    coarse_downsample = coarse_downsample,
                                    taubin_iterations = taubin_iterations,
                                    verbose = True)

            mesh_copy.vertices = torch.from_numpy(trimesh.vertices).float()
            mesh_copy.faces = torch.from_numpy(trimesh.faces).int()
        
            del trimesh
        
        return (mesh_copy,)
        
class Trellis2UnloadAllModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,)
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output_1",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1):
        print('Unloading all models ...')
        if hasattr(mm, 'current_loaded_models'):
            # Iterate backwards to safely remove items
            for i in range(len(mm.current_loaded_models) - 1, -1, -1):
                loaded_model = mm.current_loaded_models[i]
                
                print(f"[AbsoluteUnload] Force-killing: {loaded_model.model.model.__class__.__name__}")
                
                # Force VRAM unload
                loaded_model.model_unload(1e32)
                
                # Force System RAM unpinning (This is what the standard loop skipped)
                if hasattr(loaded_model.model, 'partially_unload_ram'):
                    loaded_model.model.partially_unload_ram(1e32)
                    
            # Clear ComfyUI's intermediate cross-attention and tensor caches
            if hasattr(mm, 'current_loaded_models'):
                mm.current_loaded_models.clear()            

            import comfy.controlnet
            if hasattr(comfy.controlnet, 'controlnet_loaded_models'):
                comfy.controlnet.controlnet_loaded_models.clear() 
                
        mm.free_memory(memory_required = 1e30, 
                       device = mm.get_torch_device(),
                       ram_required = 1e30)
                       
        print('Clearing cache ...')
        mm.soft_empty_cache()

        gc.collect()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print('Memory cleared')
        
        return (input_1,)    

class Trellis2SparseGeneratorWithReconViaGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "sparse_structure_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("COORDS", "INT", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("coords", "sparse_structure_resolution", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, images, seed, 
        # sparse
        sparse_structure_steps, 
        sparse_structure_guidance_strength, 
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        sparse_structure_sampler,
        sparse_structure_resolution,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        fill_holes,
        hole_iterations,
        hole_fill_algorithm,
        keep_only_shell
        ):
        
        self.seed_all(seed)
        
        self.load_vggt_model(pipeline)
        
        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}                    

        args = pipeline._pretrained_args
        sparse_sampler_prefix = pipeline.GetSamplerName(sparse_structure_sampler)
        pipeline.sparse_structure_sampler = getattr(samplers, f"Flow{sparse_sampler_prefix}GuidanceIntervalSampler")(**args['sparse_structure_sampler']['args'])
        pipeline.load_sparse_structure_vggt_model()
        pipeline.load_sparse_structure_vggt_cond()
        
        if images.ndim == 3:
            images = images.unsqueeze(0)
        
        coords = self._run_ss_stage_direct(pipeline = pipeline, 
                                           images = images, 
                                           target_ss_res = sparse_structure_resolution, 
                                           ss_sampler_params = sparse_structure_sampler_params, 
                                           verbose = verbose, 
                                           dino_lock = dino_lock, 
                                           dino_substeps = dino_substeps, 
                                           dino_foundation_cap = dino_foundation_cap, 
                                           fill_holes = fill_holes, 
                                           hole_iterations = hole_iterations, 
                                           hole_fill_algorithm = hole_fill_algorithm, 
                                           keep_only_shell = keep_only_shell)
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_sparse_structure_vggt_model()            
            pipeline.unload_sparse_structure_vggt_cond()
            self.unload_vggt_model(pipeline)

        return (coords, sparse_structure_resolution, pipeline)
        
    def load_vggt_model(self, pipeline):
        if pipeline.VGGT_model is None:
            from .vggt.vggt.models.vggt import VGGT
            pipeline.VGGT_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16            
            model_path = os.path.join(folder_paths.models_dir,'recongenvia')
            pipeline.VGGT_model = VGGT.from_pretrained(model_path)
            pipeline.VGGT_model.to('cuda')
            del pipeline.VGGT_model.depth_head
            del pipeline.VGGT_model.track_head
            pipeline.VGGT_model.eval()
                
            self._init_image_cond_model(pipeline)
            
        
    def unload_vggt_model(self, pipeline):
        del pipeline.VGGT_model
        pipeline.VGGT_model = None
        
        del pipeline.models['image_cond_model_vggt']
        pipeline.models['image_cond_model_vggt'] = None
        pipeline.image_cond_model_transform = None          
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()        
        
        
    def seed_all(self, seed: int = 0):
        import random
        """
        Set random seeds of all components.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    @torch.no_grad()
    def _run_ss_stage_direct(
        self,
        pipeline,
        images,
        target_ss_res: int,
        ss_sampler_params: dict,
        verbose: bool,
        dino_lock: float,
        dino_substeps: int,
        dino_foundation_cap: float,
        fill_holes: bool,
        hole_iterations: int,
        hole_fill_algorithm: str,
        hole_structure: int = 1,
        keep_only_shell: bool = True        
    ) -> torch.Tensor:
        """
        Run only ReconViaGen's sparse structure diffusion stage to obtain coords
        directly, without proceeding to the SLAT/mesh stage.

        Returns:
            coords : (N, 4) int tensor  [batch_idx, x, y, z]  in [0, target_ss_res)
        """           
            
        cuda_device = torch.device('cuda')

        if pipeline.low_vram:
            pipeline.VGGT_model.to(cuda_device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=pipeline.VGGT_dtype):
                aggregated_tokens_list, _ = self.vggt_feat(pipeline, images)
            b, n, _, _ = aggregated_tokens_list[0].shape
            image_cond = self.encode_image(pipeline, images).reshape(b, n, -1, 1024)
            ss_cond = self.get_ss_cond(pipeline, image_cond[:, :, 5:], aggregated_tokens_list, 1)

        ss_flow_model = pipeline.models['sparse_structure_flow_vggt_model']
        sampler_params = {**pipeline.sparse_structure_sampler_params, **ss_sampler_params}
        reso = ss_flow_model.resolution
        ss_noise = torch.randn(1, ss_flow_model.in_channels, reso, reso, reso).to(cuda_device)

        with torch.autocast('cuda', dtype=torch.float16):
            ss_latent = pipeline.sparse_structure_sampler.sample(
                ss_flow_model,
                ss_noise,
                **ss_cond,
                **sampler_params,
                verbose=verbose,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap
            ).samples

        decoder = pipeline.models['sparse_structure_decoder']
        decoded = decoder(ss_latent) > 0
        if target_ss_res != decoded.shape[2]:
            ratio = decoded.shape[2] // target_ss_res
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        
        # Optionally fill holes in the sparse voxel grid using the selected algorithm
        if fill_holes:
            try:
                from scipy.ndimage import binary_closing, label, binary_fill_holes
                arr = decoded.cpu().numpy()
                if arr.ndim == 5:
                    arr = arr[:, 0]
                closed = np.zeros_like(arr)
                for b in range(arr.shape[0]):
                    filled = arr[b].astype(np.bool_)
                    inv = ~filled
                    labeled, num_features = label(inv)
                    border_mask = np.zeros_like(inv)
                    border_mask[0,:,:] = border_mask[-1,:,:] = 1
                    border_mask[:,0,:] = border_mask[:,-1,:] = 1
                    border_mask[:,:,0] = border_mask[:,:,-1] = 1
                    border_labels = np.unique(labeled[border_mask==1])
                    holes = np.isin(labeled, border_labels, invert=True) & (labeled > 0)
                    n_holes = np.unique(labeled[holes]).size
                    print(f"[Sparse HoleFill] Batch {b}: Found {n_holes} holes before filling.")
                    if hole_fill_algorithm == "morphological_closing":
                        closed[b] = binary_closing(arr[b], structure=np.ones((hole_structure,)*3), iterations=hole_iterations)
                    elif hole_fill_algorithm == "flood_fill":
                        # Robust structure-preserving hole filling:
                        # 1. Morphological closing to connect small gaps
                        # 2. Fill internal holes
                        # 3. Keep only the largest connected component
                        from scipy.ndimage import binary_closing, label, binary_fill_holes
                        # Step 1: Morphological closing (small structure, 1 iter)
                        closed1 = binary_closing(arr[b], structure=np.ones((hole_structure,)*3), iterations=hole_iterations)
                        # Step 2: Fill internal holes
                        filled = binary_fill_holes(closed1)
                        # Step 3: Keep only the largest connected component
                        labeled, num = label(filled)
                        if num > 0:
                            sizes = np.bincount(labeled.ravel())
                            sizes[0] = 0  # background
                            largest = sizes.argmax()
                            closed[b] = (labeled == largest)
                        else:
                            closed[b] = filled                      
                    elif hole_fill_algorithm == "remove_small_holes":
                        # Remove small holes by area (2D slices)
                        from skimage.morphology import remove_small_holes
                        # Apply per-slice (z axis)
                        temp = np.copy(arr[b])
                        for z in range(temp.shape[0]):
                            temp[z] = remove_small_holes(temp[z].astype(bool), area_threshold=hole_structure**2)
                        closed[b] = temp
                    else:
                        print(f"[Sparse HoleFill] Unknown algorithm: {hole_fill_algorithm}, skipping.")
                        closed[b] = arr[b]
                    # Count holes after filling
                    filled2 = closed[b].astype(np.bool_)
                    inv2 = ~filled2
                    labeled2, num_features2 = label(inv2)
                    border_labels2 = np.unique(labeled2[border_mask==1])
                    holes2 = np.isin(labeled2, border_labels2, invert=True) & (labeled2 > 0)
                    n_holes2 = np.unique(labeled2[holes2]).size
                    print(f"[Sparse HoleFill] Batch {b}: {n_holes-n_holes2} holes filled, {n_holes2} remain after filling.")

                    # Optionally remove deeply interior voxels, keeping surface and near-surface structure
                    if keep_only_shell:
                        from scipy.ndimage import binary_erosion

                        filled = closed[b].astype(np.bool_)
                        before_count = int(filled.sum())
                        struct = np.ones((3, 3, 3), dtype=bool)
                        # Erode twice: only voxels surviving 2 erosion passes are >=2 layers deep
                        # This preserves thin structures (e.g. necks with 3x3 cross-section)
                        eroded = binary_erosion(filled, structure=struct, border_value=0)
                        eroded = binary_erosion(eroded, structure=struct, border_value=0)
                        # Remove only deeply interior voxels (>=2 layers from any surface)
                        shell = filled & ~eroded
                        closed[b] = shell
                        after_count = int(shell.sum())
                        if verbose:
                            print(f"[Sparse Shell] Batch {b}: {before_count} -> {after_count} voxels (removed {before_count - after_count} deeply interior)")

                decoded = torch.from_numpy(closed).to(decoded.device)


                # Debug: print tensor info before extracting coordinates
                if verbose:
                    print(f"[Sparse HoleFill] decoded shape: {decoded.shape}, dtype: {decoded.dtype}, device: {decoded.device}")
                    print(f"[Sparse HoleFill] decoded min: {decoded.min().item()}, max: {decoded.max().item()}, unique: {torch.unique(decoded)}")
                # Safety: ensure tensor is contiguous and on CPU for argwhere
                decoded = decoded.contiguous().cpu()

            except ImportError:
                print("[Warning] scipy or skimage not installed, skipping hole filling.")
            except Exception as e:
                print(f"[Warning] Hole filling failed: {e}")

            try:
                coords = torch.argwhere(decoded)[:, [0, 1, 2, 3]].int()
            except Exception as e:
                print(f"[Sparse HoleFill] Error in torch.argwhere: {e}")
                raise

            if verbose:
                print(f"[Sparse HoleFill] coords shape: {coords.shape}, min: {coords.min(dim=0).values.tolist() if coords.numel()>0 else 'empty'}, max: {coords.max(dim=0).values.tolist() if coords.numel()>0 else 'empty'}")

            if coords.numel() == 0:
                raise RuntimeError("No voxels remain after hole filling/shell extraction. The mask is empty. Adjust your input, mask, or hole filling parameters.")
        else:
            coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        if pipeline.low_vram:
            pipeline.VGGT_model.to('cpu')
            decoder.to('cpu')
            ss_cond = pipeline._cond_cpu(ss_cond)
            torch.cuda.empty_cache()

        return coords
        
        
    @torch.no_grad()
    def _run_ss_stage(
        self,
        pipeline,
        images,
        target_ss_res: int,
        ss_sampler_params: dict,
        slat_sampler_params: dict,
    ) -> torch.Tensor:
        """
        Generate a rough mesh via vggt_pipeline, then voxelise it into
        surface-only coords at target_ss_res^3 for the downstream shape/tex stages.

        Returns:
            coords : (N, 4) int tensor  [batch_idx, x, y, z]  in [0, target_ss_res)
        """
        vp = self.vggt_pipeline
        # vp.device is dynamic (inferred from model params), so when models are on
        # CPU it returns 'cpu'. Hardcode the target cuda device instead.
        cuda_device = torch.device('cuda')

        if self.low_vram:
            self._vggt_models_to(cuda_device)

        outputs, _, _ = vp.run(
            image=images,
            formats=["mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params=ss_sampler_params,
            slat_sampler_params=slat_sampler_params,
        )
        mesh_result = outputs["mesh"][0]
        coords = self._mesh_to_surface_coords(mesh_result, target_ss_res, cuda_device)

        if self.low_vram:
            self._vggt_models_to('cpu')
            torch.cuda.empty_cache()

        return coords        

    @torch.no_grad()
    def vggt_feat(self, pipeline, image):
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, H, W, C) or (B, C, H, W)"
            # ComfyUI IMAGE tensors are (B, H, W, C); convert to (B, C, H, W)
            if image.shape[-1] in (3, 4):
                image = image.permute(0, 3, 1, 2)
            image = F.interpolate(image, 518, mode='bilinear', align_corners=False)
            image = image.to(pipeline.device)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(pipeline.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=pipeline.VGGT_dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                aggregated_tokens_list, _ = pipeline.VGGT_model.aggregator(image[None])

        return aggregated_tokens_list, image

    def get_ss_cond(self, pipeline, image_cond: torch.Tensor, aggregated_tokens_list: list, num_samples: int) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        pipeline.models['sparse_structure_vggt_cond'].to(pipeline.device)
        cond = pipeline.models['sparse_structure_vggt_cond'](aggregated_tokens_list, image_cond)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def get_slat_cond(self, pipeline, image_cond: torch.Tensor, aggregated_tokens_list: list, num_samples: int) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        b, n, _, _ = aggregated_tokens_list[0].shape
        cond = pipeline.models['slat_vggt_cond'](aggregated_tokens_list, image_cond).reshape(b, n, -1, 1024)
        cond = [c.squeeze(1) for c in cond.split(1, dim=1)]
        neg_cond = [torch.zeros_like(c) for c in cond]
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }     

    @torch.no_grad()
    def encode_image(self, pipeline, image, w_layernorm=True) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, H, W, C) or (B, C, H, W)"
            # ComfyUI IMAGE tensors are (B, H, W, C); convert to (B, C, H, W)
            if image.shape[-1] in (3, 4):
                image = image.permute(0, 3, 1, 2)
            image = F.interpolate(image, 518, mode='bilinear', align_corners=False)
            image = image.to(pipeline.device)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(pipeline.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = pipeline.image_cond_model_transform(image).to(pipeline.device)
        pipeline.models['image_cond_model_vggt'].to(pipeline.device)
        features = pipeline.models['image_cond_model_vggt'](image, is_training=True)['x_prenorm']
        if w_layernorm:
            features = F.layer_norm(features, features.shape[-1:])
        return features

    def _init_image_cond_model(self, pipeline, name: str = "dinov2_vitl14_reg"):
        """
        Initialize the image conditioning model.
        """
        try:
            dinov2_model = torch.hub.load(os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main'), name, source='local',pretrained=True)
        except:
            dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        pipeline.models['image_cond_model_vggt'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pipeline.image_cond_model_transform = transform  

class Trellis2ExtractImagesFromVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_file": ("STRING",),
                "frames_per_second": ("INT",{"default":1,"min":1,"max":50,"step":1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, video_file, frames_per_second):
        import imageio
        
        vid = imageio.get_reader(video_file, 'ffmpeg')
        fps = vid.get_meta_data()['fps']
        frames = []
        for i, frame in enumerate(vid):
            if i % max(int(fps/frames_per_second), 1) == 0:
                img = Image.fromarray(frame)
                W, H = img.size
                img = img.resize((int(W / H * 1024), 1024))
                frames.append(img)
        vid.close()
        
        tensor_list = [torch.from_numpy(np.array(img).astype(np.float32) / 255.0) for img in frames]
            
        print(f"{len(frames)} frames extracted")
        
        tensor_frames = torch.stack(tensor_list)
        #tensor_frames = tensor_frames.permute(0, 2, 3, 1)
        
        return (tensor_frames,)
        
class Trellis2MaxTokensCalculator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "default_max_tokens": ("INT",{"default":999999,"min":1,"max":999999,"step":1}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("max_tokens",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, default_max_tokens):
        device = mm.get_torch_device()
        try:
            total_vram_bytes = mm.get_total_memory(device)
            total_vram_gb = total_vram_bytes / (1024 ** 3)
            max_tokens = int(2600 * total_vram_gb)
        except Exception as e:
            print(f"Error, cannot get VRAM size : {e}")
            max_tokens = default_max_tokens
        
        return (max_tokens,)  
        
class Trellis2ImageCondMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "front_image": ("IMAGE",)
            },
            "optional": {
                "back_image": ("IMAGE",),
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
            },            
        }

    RETURN_TYPES = ("IMAGE_CONDS", "IMAGE_CONDS", "VIEWS_LIST", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("conds_512", "conds_1024", "views_list", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, 
        pipeline, 
        front_image, 
        back_image = None,
        left_image = None,
        right_image = None):

        front_pil = tensor2pil(front_image)
        
        # Convert optional view image tensors to PIL
        back_pil = tensor2pil(back_image) if back_image is not None else None
        left_pil = tensor2pil(left_image) if left_image is not None else None
        right_pil = tensor2pil(right_image) if right_image is not None else None           
            
        # Collect views
        views_dict = {'front': front_pil}
        if back_pil is not None: views_dict['back'] = back_pil
        if left_pil is not None: views_dict['left'] = left_pil
        if right_pil is not None: views_dict['right'] = right_pil
        
        views_list = list(views_dict.keys())            
        
        # Calculate conditioning per view
        conds_512 = {}
        conds_1024 = {}
        
        pipeline.load_image_cond_model() 

        for v, img in views_dict.items():
            c512 = pipeline.get_cond([img], 512)
            c1024 = pipeline.get_cond([img], 1024)
            conds_512[v] = c512
            conds_1024[v] = c1024           
            
        if not pipeline.keep_models_loaded:
            pipeline.unload_image_cond_model()              

        return (conds_512, conds_1024, views_list, pipeline,)          

class Trellis2SparseMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_conds": ("IMAGE_CONDS",),
                "views_list": ("VIEWS_LIST",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "sparse_structure_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("COORDS", "INT", "VIEWS_LIST", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("coords", "sparse_structure_resolution", "views_list", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_conds, views_list, seed, 
        # sparse
        sparse_structure_steps, 
        sparse_structure_guidance_strength, 
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        sparse_structure_sampler,
        sparse_structure_resolution,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,
        fill_holes,
        hole_iterations,
        verbose,
        dino_lock,
        dino_substeps,
        hole_fill_algorithm,
        dino_foundation_cap,
        keep_only_shell,
        front_axis,
        blend_temperature
        ):
        
        self.seed_all(seed)
        
        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}                    

        sparse_sampler_prefix = pipeline.GetSamplerName(sparse_structure_sampler)
        pipeline.load_sparse_structure_model()                
            
        if pipeline.low_vram:
            for v in image_conds:
                image_conds[v] = pipeline._cond_to(image_conds[v], pipeline.device)
                
        # Sample sparse structure latent
        flow_model = pipeline.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(1, in_channels, reso, reso, reso).to(pipeline.device)
        
        sampler_class = getattr(samplers, f"Flow{sparse_sampler_prefix}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        sampler = sampler_class(
            sigma_min=1e-5,
            resolution=flow_model.resolution if hasattr(flow_model, 'resolution') else flow_model[0].resolution
        )        
        
        sparse_structure_sampler_params = {**pipeline.sparse_structure_sampler_params, **sparse_structure_sampler_params}
        
        if pipeline.low_vram:
            flow_model.to(pipeline.device)          
            
        z_s = sampler.sample(
            flow_model,
            noise,
            conds=image_conds,            
            **sparse_structure_sampler_params,            
            views=views_list,
            front_axis=front_axis,
            blend_temperature=blend_temperature,            
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            tqdm_desc="Sampling sparse structure (MultiView)",
        ).samples
        
        if pipeline.low_vram:
            flow_model.cpu()
            pipeline._cleanup_cuda()
            
        # Decode sparse structure latent
        decoder = pipeline.models['sparse_structure_decoder']
        if pipeline.low_vram:
            decoder.to(pipeline.device)
            
        # Standard decoding logic from sample_sparse_structure
        decoded = decoder(z_s) > 0
        
        if pipeline.low_vram:
            decoder.cpu()
            pipeline._cleanup_cuda()
            
        # if resolution != decoded.shape[2]:
            # ratio = decoded.shape[2] // resolution
            # decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        if sparse_structure_resolution != decoded.shape[2]:
            if sparse_structure_resolution < decoded.shape[2]:
                ratio = decoded.shape[2] // sparse_structure_resolution
                decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
            else:
                decoded = torch.nn.functional.interpolate(decoded.float(), size=(sparse_structure_resolution, sparse_structure_resolution, sparse_structure_resolution), mode='nearest') > 0.5            

        # Optionally fill holes in the sparse voxel grid using the selected algorithm
        if fill_holes:
            hole_structure = 1
            try:
                from scipy.ndimage import binary_closing, label, binary_fill_holes
                arr = decoded.cpu().numpy()
                if arr.ndim == 5:
                    arr = arr[:, 0]
                closed = np.zeros_like(arr)
                for b in range(arr.shape[0]):
                    filled = arr[b].astype(np.bool_)
                    inv = ~filled
                    labeled, num_features = label(inv)
                    border_mask = np.zeros_like(inv)
                    border_mask[0,:,:] = border_mask[-1,:,:] = 1
                    border_mask[:,0,:] = border_mask[:,-1,:] = 1
                    border_mask[:,:,0] = border_mask[:,:,-1] = 1
                    border_labels = np.unique(labeled[border_mask==1])
                    holes = np.isin(labeled, border_labels, invert=True) & (labeled > 0)
                    n_holes = np.unique(labeled[holes]).size
                    print(f"[Sparse HoleFill] Batch {b}: Found {n_holes} holes before filling.")
                    if hole_fill_algorithm == "morphological_closing":
                        closed[b] = binary_closing(arr[b], structure=np.ones((hole_structure,)*3), iterations=hole_iterations)
                    elif hole_fill_algorithm == "flood_fill":
                        # Robust structure-preserving hole filling:
                        # 1. Morphological closing to connect small gaps
                        # 2. Fill internal holes
                        # 3. Keep only the largest connected component
                        from scipy.ndimage import binary_closing, label, binary_fill_holes
                        # Step 1: Morphological closing (small structure, 1 iter)
                        closed1 = binary_closing(arr[b], structure=np.ones((hole_structure,)*3), iterations=hole_iterations)
                        # Step 2: Fill internal holes
                        filled = binary_fill_holes(closed1)
                        # Step 3: Keep only the largest connected component
                        labeled, num = label(filled)
                        if num > 0:
                            sizes = np.bincount(labeled.ravel())
                            sizes[0] = 0  # background
                            largest = sizes.argmax()
                            closed[b] = (labeled == largest)
                        else:
                            closed[b] = filled                      
                    elif hole_fill_algorithm == "remove_small_holes":
                        # Remove small holes by area (2D slices)
                        from skimage.morphology import remove_small_holes
                        # Apply per-slice (z axis)
                        temp = np.copy(arr[b])
                        for z in range(temp.shape[0]):
                            temp[z] = remove_small_holes(temp[z].astype(bool), area_threshold=hole_structure**2)
                        closed[b] = temp
                    else:
                        print(f"[Sparse HoleFill] Unknown algorithm: {hole_fill_algorithm}, skipping.")
                        closed[b] = arr[b]
                    # Count holes after filling
                    filled2 = closed[b].astype(np.bool_)
                    inv2 = ~filled2
                    labeled2, num_features2 = label(inv2)
                    border_labels2 = np.unique(labeled2[border_mask==1])
                    holes2 = np.isin(labeled2, border_labels2, invert=True) & (labeled2 > 0)
                    n_holes2 = np.unique(labeled2[holes2]).size
                    print(f"[Sparse HoleFill] Batch {b}: {n_holes-n_holes2} holes filled, {n_holes2} remain after filling.")

                    # Optionally remove deeply interior voxels, keeping surface and near-surface structure
                    if keep_only_shell:
                        from scipy.ndimage import binary_erosion

                        filled = closed[b].astype(np.bool_)
                        before_count = int(filled.sum())
                        struct = np.ones((3, 3, 3), dtype=bool)
                        # Erode twice: only voxels surviving 2 erosion passes are >=2 layers deep
                        # This preserves thin structures (e.g. necks with 3x3 cross-section)
                        eroded = binary_erosion(filled, structure=struct, border_value=0)
                        eroded = binary_erosion(eroded, structure=struct, border_value=0)
                        # Remove only deeply interior voxels (>=2 layers from any surface)
                        shell = filled & ~eroded
                        closed[b] = shell
                        after_count = int(shell.sum())
                        if verbose:
                            print(f"[Sparse Shell] Batch {b}: {before_count} -> {after_count} voxels (removed {before_count - after_count} deeply interior)")

                decoded = torch.from_numpy(closed).to(decoded.device)


                # Debug: print tensor info before extracting coordinates
                if verbose:
                    print(f"[Sparse HoleFill] decoded shape: {decoded.shape}, dtype: {decoded.dtype}, device: {decoded.device}")
                    print(f"[Sparse HoleFill] decoded min: {decoded.min().item()}, max: {decoded.max().item()}, unique: {torch.unique(decoded)}")
                # Safety: ensure tensor is contiguous and on CPU for argwhere
                decoded = decoded.contiguous().cpu()

            except ImportError:
                print("[Warning] scipy or skimage not installed, skipping hole filling.")
            except Exception as e:
                print(f"[Warning] Hole filling failed: {e}")

            try:
                coords = torch.argwhere(decoded)[:, [0, 1, 2, 3]].int()
            except Exception as e:
                print(f"[Sparse HoleFill] Error in torch.argwhere: {e}")
                raise

            if verbose:
                print(f"[Sparse HoleFill] coords shape: {coords.shape}, min: {coords.min(dim=0).values.tolist() if coords.numel()>0 else 'empty'}, max: {coords.max(dim=0).values.tolist() if coords.numel()>0 else 'empty'}")

            if coords.numel() == 0:
                raise RuntimeError("No voxels remain after hole filling/shell extraction. The mask is empty. Adjust your input, mask, or hole filling parameters.")
        else:
            coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        coords = coords.cpu()

        del decoded
        del z_s
        if pipeline.low_vram:
            for v in image_conds:
                image_conds[v] = pipeline._cond_cpu(image_conds[v])
            pipeline._cleanup_cuda()

        return (coords, sparse_structure_resolution, views_list, pipeline,)
        
    def seed_all(self, seed: int = 0):
        import random
        """
        Set random seeds of all components.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   

class Trellis2ShapeMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_conds": ("IMAGE_CONDS",),
                "views_list": ("VIEWS_LIST",),
                "coords": ("COORDS",),
                "resolution": ([512,1024],{"default":1024}),                
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),  
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),                
            },
        }

    RETURN_TYPES = ("SHAPE_SLAT", "INT", "VIEWS_LIST", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("shape_slat", "resolution", "views_list", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_conds, views_list, coords, resolution,      
        # shape
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,
        shape_sampler,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        front_axis,
        blend_temperature
        ):
            
        shape_guidance_interval = [shape_guidance_interval_start, shape_guidance_interval_end]        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}            
        
        if resolution == 512:
             pipeline.load_shape_slat_flow_model_512()
             shape_slat = self.sample(
                pipeline, shape_sampler,
                image_conds, views_list,
                pipeline.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                verbose=verbose,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap
             )
             if not pipeline.keep_models_loaded:
                 pipeline.unload_shape_slat_flow_model_512()
                 
        elif resolution == 1024:
             pipeline.load_shape_slat_flow_model_1024()
             shape_slat = self.sample(
                pipeline, shape_sampler,
                image_conds, views_list,
                pipeline.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                verbose=verbose,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap
             )
             if not pipeline.keep_models_loaded:
                 pipeline.unload_shape_slat_flow_model_1024()
        
        return (shape_slat, resolution, views_list, pipeline,)
        
    def sample(
        self,
        pipeline,
        shape_sampler,
        conds: dict,
        views: list,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        verbose: bool = False,
        dino_lock: float = 0.00,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92
    ) -> SparseTensor:
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_to(conds[v], pipeline.device)

        coords_dev = coords.to(pipeline.device)                         
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=pipeline.device),
            coords=coords_dev,
        )
        
        # sampler = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            # sigma_min=1e-5,
            # resolution=flow_model.resolution,
        # )
        sampler_class = getattr(samplers, f"Flow{shape_sampler}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        sampler = sampler_class(
            sigma_min=1e-5,
            resolution=flow_model.resolution if hasattr(flow_model, 'resolution') else flow_model[0].resolution
        )        
        
        sampler_params = {**pipeline.shape_slat_sampler_params, **sampler_params}
        
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
            
        slat = sampler.sample(
            flow_model,
            noise,
            conds=conds,            
            **sampler_params,            
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,            
            verbose=verbose,
            dino_lock = dino_lock,
            dino_substeps = dino_substeps,
            dino_foundation_cap = dino_foundation_cap,
            tqdm_desc="Sampling shape SLat (MultiView)",
        ).samples
        
        if pipeline.low_vram:
            flow_model.cpu()
            pipeline._cleanup_cuda()                                

        std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_cpu(conds[v])
            pipeline._cleanup_cuda()

        return slat        
        
class Trellis2ShapeCascadeMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_conds": ("IMAGE_CONDS",),
                "views_list": ("VIEWS_LIST",),
                "shape_slat": ("SHAPE_SLAT",),
                "from_resolution": ("INT",),
                "to_resolution": ([1024,1536],{"default":1024}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),                   
            },
        }

    RETURN_TYPES = ("SHAPE_SLAT","INT","VIEWS_LIST", "TRELLIS2PIPELINE","INT",)
    RETURN_NAMES = ("shape_slat","resolution","views_list", "pipeline","num_tokens")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_conds, views_list, shape_slat, from_resolution, to_resolution, sparse_structure_resolution, max_num_tokens,      
        # shape
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,
        shape_sampler,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        front_axis,
        blend_temperature
        ):
            
        shape_guidance_interval = [shape_guidance_interval_start, shape_guidance_interval_end]        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}                    
        
        args = pipeline._pretrained_args
        shape_sampler_prefix = pipeline.GetSamplerName(shape_sampler)
        
        pipeline.load_shape_slat_flow_model_1024()         
        flow_model = pipeline.models['shape_slat_flow_model_1024']
        
        sampler_class = getattr(samplers, f"Flow{shape_sampler_prefix}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        sampler = sampler_class(
            sigma_min=1e-5,
            resolution=flow_model.resolution if hasattr(flow_model, 'resolution') else flow_model[0].resolution
        )
        
        pipeline.shape_slat_sampler = sampler

        slat, hr_resolution, num_tokens = self.sample(pipeline, shape_slat, from_resolution, to_resolution, sparse_structure_resolution, max_num_tokens, image_conds, shape_slat_sampler_params, flow_model, verbose, dino_lock, dino_substeps, dino_foundation_cap, views_list, front_axis, blend_temperature)
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_shape_slat_flow_model_1024()              
        
        return (slat, hr_resolution, views_list, pipeline, num_tokens,)         
        
    def sample(self, pipeline, slat, lr_resolution, resolution, sparse_structure_resolution, max_num_tokens, conds, sampler_params, flow_model, verbose, dino_lock, dino_substeps, dino_foundation_cap, views, front_axis, blend_temperature):
        # Upsample       
        pipeline.load_shape_slat_decoder()
        if pipeline.low_vram:
            pipeline.models['shape_slat_decoder'].to(pipeline.device)
            pipeline.models['shape_slat_decoder'].low_vram = True
        hr_coords = pipeline.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if pipeline.low_vram:
            pipeline.models['shape_slat_decoder'].cpu()
            pipeline.models['shape_slat_decoder'].low_vram = False
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_shape_slat_decoder()
        
        hr_resolution = resolution
        ratio = (sparse_structure_resolution / 32)
        
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / (lr_resolution * ratio) * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                print(f"Num Tokens: {num_tokens}")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 512
                break
                
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_to(conds[v], pipeline.device)               
        
        coords_dev = coords.to(pipeline.device)                                           
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=pipeline.device),
            coords=coords_dev,
        )
        sampler_params = {**pipeline.shape_slat_sampler_params, **sampler_params}
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
        slat = pipeline.shape_slat_sampler.sample(
            flow_model,
            noise,
            conds=conds,            
            **sampler_params,            
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,            
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            tqdm_desc="Sampling shape SLat (MultiView HR)",
        ).samples
        
        if pipeline.low_vram:
            flow_model.cpu()
            pipeline._cleanup_cuda()                                

        std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_cpu(conds[v])
            pipeline._cleanup_cuda()

        return slat, hr_resolution, num_tokens         
        
class Trellis2TexSlatMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_conds": ("IMAGE_CONDS",),
                "views_list": ("VIEWS_LIST",),
                "shape_slat": ("SHAPE_SLAT",),
                "resolution": ([512,1024],{"default":1024}),                
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),         
                "texture_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),                                                               
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),                    
            },
        }

    RETURN_TYPES = ("TEXTURE_SLAT", "VIEWS_LIST", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("texture_slat", "views_list", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_conds, views_list, shape_slat, resolution,      
        # shape
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale,
        texture_rescale_t,
        texture_sampler,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        front_axis,
        blend_temperature
        ):

        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
        
        if resolution == 512:
            pipeline.load_tex_slat_flow_model_512()
            flow_model = pipeline.models['tex_slat_flow_model_512']
        else:
            pipeline.load_tex_slat_flow_model_1024()
            flow_model = pipeline.models['tex_slat_flow_model_1024']
        
        tex_slat = self.sample(
            pipeline,
            texture_sampler,
            image_conds, views_list,
            shape_slat=shape_slat, 
            flow_model=flow_model,
            sampler_params=tex_slat_sampler_params,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap
        )  
         
        if not pipeline.keep_models_loaded:
            if resolution == 512:
                pipeline.unload_tex_slat_flow_model_512()
            else:
                pipeline.unload_tex_slat_flow_model_1024()
        
        return (tex_slat, views_list, pipeline,) 

    def sample(
        self,
        pipeline,
        sampler,
        conds: dict,
        views: list,
        shape_slat: SparseTensor,
        flow_model,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        verbose: bool = False,
        dino_lock: float = 0.00,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92
    ) -> SparseTensor:
        """
        Sample structured latent for texture with multi-view blending.
        """
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_to(conds[v], pipeline.device)

        # Normalize shape slat for conditioning
        std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat_normalized = (shape_slat - mean) / std

        #coords = shape_slat.coords
        #coords_dev = coords.to(pipeline.device)
        
        # Calculate noise channels: total input - concat cond channels
        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise_channels = in_channels - shape_slat.feats.shape[1]
        
        # noise = SparseTensor(
            # feats=torch.randn(coords.shape[0], noise_channels, device=pipeline.device),
            # coords=coords_dev,
        # )
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(pipeline.device))
        
        sampler_params = {**pipeline.tex_slat_sampler_params, **sampler_params}
        
        # sampler = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            # sigma_min=1e-5,
            # resolution=flow_model.resolution,
        # )
        tex_sampler_prefix = pipeline.GetSamplerName(sampler)
        
        sampler_class = getattr(samplers, f"Flow{tex_sampler_prefix}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        sampler = sampler_class(
            sigma_min=1e-5,
            resolution=flow_model.resolution if hasattr(flow_model, 'resolution') else flow_model[0].resolution
        )          
        
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
            
        slat = sampler.sample(
            flow_model,
            noise,
            conds=conds,
            **sampler_params,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            concat_cond=shape_slat_normalized,            
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            tqdm_desc="Sampling texture SLat (MultiView)",
        ).samples
        
        if pipeline.low_vram:
            flow_model.cpu()
            pipeline._cleanup_cuda()

        std = torch.tensor(pipeline.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(pipeline.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        #del coords_dev
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_cpu(conds[v])
            pipeline._cleanup_cuda()
            
        return slat
        
class Trellis2MoGeCameraConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "camera_angle_x": ("FLOAT",{"default":1.0000000000000000,"min":0.0000000000000000,"max":9.0000000000000000}),                
                "distance": ("FLOAT",{"default":1.0000000000000000,"min":0.0000000000000000,"max":9.9999999999999999}),
                "mesh_scale": ("FLOAT",{"default":1.00,"min":0.01,"max":9.99}),
            },
        }

    RETURN_TYPES = ("MOGE_CAM_CONFIG",)
    RETURN_NAMES = ("moge_camera_config",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, camera_angle_x, distance, mesh_scale):
        cam_config = {'camera_angle_x':camera_angle_x,'distance':distance,'mesh_scale':mesh_scale}
        
        return (cam_config,)   

class Trellis2FovMoGeCameraConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fov": ("FLOAT",{"default":0.200,"min":0.001,"max":359.999, "step":0.001}),
                "unit": (["deg","rad"],{"default":"rad"}),
                "extend_pixel": ("INT",{"default":0,"min":0,"max":1000}),
                "mesh_scale": ("FLOAT",{"default":1.0,"min":0.1,"max":9.9,"step":0.1}),
            },
        }

    RETURN_TYPES = ("MOGE_CAM_CONFIG",)
    RETURN_NAMES = ("moge_camera_config",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, fov, unit, extend_pixel, mesh_scale):        
        from .trellis2.utils.camera import distance_from_fov
        
        image_resolution = 512
        
        if unit == "rad":
            camera_angle_x = fov
            fov_deg = math.degrees(fov)
        else:
            camera_angle_x = math.radians(fov)
            fov_deg = fov
            
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x, grid_point,
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
            mesh_scale, image_resolution
        )["distance_from_x"]
        cam_config = {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}
        print(cam_config)
        
        return (cam_config,)       

class Trellis2ImagesToViewConfigs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "azimuths": ("STRING",),
                "elevations": ("STRING",),
            },
        }

    RETURN_TYPES = ("TRELLIS2_VIEWCONFIGS",)
    RETURN_NAMES = ("view_config_list",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, images, azimuths, elevations,):
        custom_az_list = self._parse_angles(azimuths)
        custom_el_list = self._parse_angles(elevations)     
        
        view_configs = []
        for i,a,e in zip(images,custom_az_list,custom_el_list):                       
            if a<=30:
                if e>=0:
                    if e<20:
                        prompt = "Generate the front view"
                    elif e<60:
                        prompt = "Generate the top-front-view"
                    else:
                        prompt = "Generate the top view"
                else:
                    if e>-20:
                        prompt = "Generate the front view"
                    elif e>-60:
                        prompt = "Generate the bottom-front-view"
                    else:
                        prompt = "Generate the bottom view"                    
            elif a<=75:
                if e>=0:
                    if e<20:
                        prompt = "Generate the front-left view"
                    elif e<60:
                        prompt = "Generate the top-front-left view"
                    else:
                        prompt = "Generate the top-left view"
                else:
                    if e>-20:
                        prompt = "Generate the front-left view"
                    elif e>-60:
                        prompt = "Generate the bottom-front-left view"
                    else:
                        prompt = "Generate the bottom-left view"                    
            elif a<=120:
                if e>=0:
                    if e<20:
                        prompt = "Generate the left side view"
                    else:
                        prompt = "Generate the top-left side view"
                else:
                    if e>-20:
                        prompt = "Generate the left side view"
                    else:
                        prompt = "Generate the bottom-left side view"                    
            elif a<=165:
                if e>=0:
                    if e<20:
                        prompt = "Generate the back-left view"
                    elif e<60:
                        prompt = "Generate the top-back-left view"
                    else:
                        prompt = "Generate the top-left view"
                else:
                    if e>-20:
                        prompt = "Generate the back-left view"
                    elif e>-60:
                        prompt = "Generate the bottom-back-left view"
                    else:
                        prompt = "Generate the bottom-left view"
                    
            elif a<=210:
                if e>=0:
                    if e<20:
                        prompt = "Generate the back view"
                    elif e<60:
                        prompt = "Generate the top-back view"
                    else:
                        prompt = "Generate the top view"
                else:
                    if e>-20:
                        prompt = "Generate the back view"
                    elif e>-60:
                        prompt = "Generate the bottom-back view"
                    else:
                        prompt = "Generate the bottom view"                    
            elif a<=255:
                if e>=0:
                    if e<20:
                        prompt = "Generate the back-right view"
                    elif e<60:
                        prompt = "Generate the top-back-right view"
                    else:
                        prompt = "Generate the top-right view"
                else:
                    if e>-20:
                        prompt = "Generate the back-right view"
                    elif e>-60:
                        prompt = "Generate the bottom-back-right view"
                    else:
                        prompt = "Generate the bottom-right view"                    
            elif a<=300:
                if e>=0:
                    if e<20:
                        prompt = "Generate the right side view"
                    else:
                        prompt = "Generate the top-right side view"
                else:
                    if e>-20:
                        prompt = "Generate the right side view"
                    else:
                        prompt = "Generate the bottom-right side view"
                    
            else:
                if e>=0:
                    if e<20:
                        prompt = "Generate the front-right view"
                    elif e<60:
                        prompt = "Generate the top-front-right view"
                    else:
                        prompt = "Generate the top-right view"
                else:
                    if e>-20:
                        prompt = "Generate the front-right view"
                    elif e>-60:
                        prompt = "Generate the bottom-front-right view"
                    else:
                        prompt = "Generate the bottom-right view"                
            
            img_tensor = i.unsqueeze(0)
            
            view_config = ViewConfig(img_tensor,a,e,prompt)
            view_configs.append(view_config)
            
            print(f"azimuth {a} - elevation {e} : {prompt}")
            
        return (view_configs,) 

    def _parse_angles(self, angle_string):
        """Parse comma-separated angles into list of floats."""
        if not angle_string or angle_string.strip() == "":
            return []
        try:
            return [float(x.strip()) for x in angle_string.split(",") if x.strip()]
        except ValueError:
            print(f"[MultiView] Warning: Could not parse angles: {angle_string}")
            return []        

class Trellis2ViewConfigsToList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "view_config_list": ("TRELLIS2_VIEWCONFIGS",),
            },
        }

    RETURN_TYPES = ("TRELLIS2_VIEWCONFIG",)
    RETURN_NAMES = ("view_config",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, view_config_list,):
        return (view_config_list,)    

class Trellis2ExtractViewConfigDetails:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "view_config": ("TRELLIS2_VIEWCONFIG",),
            },
        }

    RETURN_TYPES = ("IMAGE","FLOAT","FLOAT","STRING",)
    RETURN_NAMES = ("image","azimuth","elevation","prompt",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, view_config,):        
        return (view_config.image, view_config.azimuth, view_config.elevation, view_config.prompt,)

class Trellis2LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "pattern": ("STRING", {"default": "*.png"}),
            },
        }

    # Added the third output: image_with_alpha
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("images", "masks", "image_with_alpha")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, folder_path, pattern):
        import glob
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder path does not exist: {folder_path}")

        search_path = os.path.join(folder_path, pattern)
        file_paths = sorted(glob.glob(search_path))

        if not file_paths:
            raise FileNotFoundError(f"No files matched the pattern '{pattern}' in '{folder_path}'")

        loaded_images = []
        loaded_masks = []
        loaded_rgba = []

        for file_path in file_paths:
            try:
                img = Image.open(file_path)
                img = ImageOps.exif_transpose(img)
                
                # Force conversion to RGBA to safely extract alpha
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                
                img_rgb = img.convert("RGB")
                alpha = img.split()[-1]
                
                # 1. Standard ComfyUI RGB Tensor -> [H, W, 3]
                image_np = np.array(img_rgb).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                
                # 2. Standard ComfyUI Mask Tensor -> [H, W]
                mask_np = np.array(alpha).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np)
                
                # 3. Combine them into an RGBA Tensor -> [H, W, 4]
                # We unsqueeze the mask from [H, W] to [H, W, 1] so it can concatenate with [H, W, 3]
                rgba_tensor = torch.cat([image_tensor, mask_tensor.unsqueeze(-1)], dim=-1)
                
                loaded_images.append(image_tensor)
                loaded_masks.append(mask_tensor)
                loaded_rgba.append(rgba_tensor)
                
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
                continue

        if not loaded_images:
            raise ValueError("No valid images could be loaded.")

        # Batch stack all outputs cleanly
        output_images = torch.stack(loaded_images, dim=0)       # [B, H, W, 3]
        output_masks = torch.stack(loaded_masks, dim=0)         # [B, H, W]
        output_rgba = torch.stack(loaded_rgba, dim=0)           # [B, H, W, 4]

        return (output_images, output_masks, output_rgba)

class Trellis2RenderMultiViewNvdiffrast:    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "render_size": ("INT", {"default": 1024, "min": 512, "max": 8192}),
                "ortho_scale": ("FLOAT", {"default": 1.2, "min": 0.05, "max": 10.0, "step": 0.01}),
                "azimuths": ("STRING",{"default":"0,90,180,270,0,0"}),
                "elevations": ("STRING",{"default":"0,0,0,0,90,-90"}),
                "add_shading": ("BOOLEAN", {"default": False, "tooltip": "A simple Lambert shading"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE","FLOAT", "STRING", "STRING",)
    RETURN_NAMES = ("images","ortho_scale", "azimuths", "elevations",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True
    
    def render_view_nvdiffrast(self, mesh, elev, azim, resolution, scale, add_shading):
        verts = torch.from_numpy(mesh.vertices).float().cuda()
        faces = torch.from_numpy(mesh.faces).int().cuda()
        uvs   = torch.from_numpy(mesh.visual.uv).float().cuda()
        
        baseColorTexture = self.getBaseColorTexture(mesh)
        tex = torch.from_numpy(np.array(baseColorTexture)).float().cuda() / 255.0

        R, T = self.build_camera_RT(azim, elev, dist=scale, device=verts.device)
        verts_cam = verts @ R.T + T
        
        verts_h = torch.cat([verts_cam, torch.ones_like(verts_cam[:, :1])], dim=1)
        
        P = self.ortho_projection(scale, verts.device)
        verts_clip = (P @ verts_h.T).T

        verts_clip = verts_clip.unsqueeze(0).contiguous()
        faces = faces.contiguous()

        glctx = dr.RasterizeGLContext()
        rast, _ = dr.rasterize(
            glctx,
            verts_clip,
            faces,
            resolution=[resolution, resolution] 
        )

        uv_attr = dr.interpolate(uvs.unsqueeze(0), rast, faces)[0]
        u = uv_attr[..., 0]
        v = uv_attr[..., 1]
        grid_x = u * 2 - 1
        grid_y = (1.0 - v) * 2 - 1 # vertical flip
        uv_grid = torch.stack([grid_x, grid_y], dim=-1)

        tex_img = tex.permute(2,0,1).unsqueeze(0)
        color = torch.nn.functional.grid_sample(
            tex_img,
            uv_grid,
            align_corners=True
        )[0]
        
        color = color.permute(1,2,0)

        if add_shading:
            # ---------------- SHADING LAMBERT ----------------
            # Per-vertex normals
            normals = torch.from_numpy(mesh.vertex_normals).float().cuda()
            normals_cam = normals @ R.T
            normals_cam = torch.nn.functional.normalize(normals_cam, dim=1)

            # Normals interpolation per-pixel
            normals_attr = dr.interpolate(normals_cam.unsqueeze(0), rast, faces)[0]  # (H,W,3)
            normals_attr = torch.nn.functional.normalize(normals_attr, dim=-1)

            # Directional light in camera space (front camera)
            light_dir = torch.tensor([0, 0, 1], device=verts.device).float()
            light_dir = torch.nn.functional.normalize(light_dir, dim=0)

            # Lambert: max(dot(n, l), 0)
            shading = torch.clamp((normals_attr * light_dir).sum(dim=-1, keepdim=True), 0, 1)  # (H,W,1)
        
            shaded_rgb = (color[..., :3] * shading).squeeze(0)

            # Apply shading only to RGB channels (keep alpha as is if present)
            if color.shape[-1] == 4:
                alpha   = color[..., 3:]
                color = torch.cat([shaded_rgb, alpha], dim=-1)
            else:
                color = shaded_rgb
        
        return color, rast
    
    def render(
        self,
        elev, azim,
        add_shading,
        resolution=None,
        tex=None,
        keep_alpha=True,
        bgcolor=None,
        return_type='th',
        scale=1.0,
        mesh=None,
    ):        
        if tex is not None:
            if isinstance(tex, Image.Image):
                tex = torch.tensor(np.array(tex) / 255.0)
            elif isinstance(tex, np.ndarray):
                tex = torch.tensor(tex)
            if tex.dim() == 2:
                tex = tex.unsqueeze(-1)
            tex = tex.float().to(self.device)
        
        image, rast = self.render_view_nvdiffrast(mesh, elev, azim, resolution, scale, add_shading)
        
        rast = rast[0]
        mask = (rast[:, :, 3:4] > 0).float() 

        if bgcolor is None:
            bgcolor = [0 for _ in range(image.shape[-1] - 1)]
        image = image * mask + (1 - mask) * torch.tensor(bgcolor + [0], device=image.device)
        if keep_alpha == False:
            image = image[..., :-1]

        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.squeeze(-1).cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        
        return image.cpu()
    
    def render_textured_multiview(self, camera_elevs, camera_azims, ortho_scale, resolution, mesh, add_shading):      
        textured_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            textured_map = self.render(
                elev, azim, 
                add_shading,
                return_type='th', 
                scale=ortho_scale, 
                resolution=resolution,
                mesh=mesh)
            textured_maps.append(textured_map)
        
        return textured_maps
    
    def process(
        self,
        trimesh,
        render_size,
        ortho_scale,
        azimuths,
        elevations,
        add_shading
    ):
        reset_cuda()
        
        if not hasattr(trimesh.visual, 'material'):
            raise Exception("Trimesh does not have a material")
                    
        custom_az_list = Trellis2RenderMultiView()._parse_angles(azimuths)
        custom_el_list = Trellis2RenderMultiView()._parse_angles(elevations)
              
        if custom_az_list and custom_el_list:
            if len(custom_az_list) != len(custom_el_list):
                raise Exception("azimuths and elevations must have the same amount of values")
                
            textured_maps = self.render_textured_multiview(custom_el_list, custom_az_list, 
                ortho_scale, render_size, trimesh, add_shading)
            custom_images = torch.stack(textured_maps, dim=0)
                        
            return (custom_images, ortho_scale, azimuths, elevations,)
        else:
            raise Exception("azimuths and elevations are required")             
        
    def get_camera_vectors(self, azimuth: float, elevation: float, device='cuda'):
        az = math.radians(azimuth)
        el = math.radians(elevation)

        # look is the negated camera-position direction, so that
        #   eye = -look*dist = dist * [cos(el)sin(az), sin(el), cos(el)cos(az)]
        # places az=0/90/180/270 -> front(+Z)/left(+X)/back(-Z)/right(-X) and
        # el=+/-90 -> top(+Y)/bottom(-Y), matching a standard turnaround.
        look = torch.tensor([
            -math.cos(el) * math.sin(az),
            -math.sin(el),
            -math.cos(el) * math.cos(az)
        ], device=device, dtype=torch.float32)
        look = look / (look.norm() + 1e-8)
        
        world_up = torch.tensor([0., 1., 0.], device=device, dtype=torch.float32)
        # If look is parallel to world_up, choose an in-plane up reference.
        # Flip it between the top (look.y < 0) and bottom poles so both views
        # come out upright instead of one being rolled 180 degrees.
        if torch.abs(torch.dot(look, world_up)) > 0.99:
            up_z = -1.0 if look[1] < 0 else 1.0
            world_up = torch.tensor([0., 0., up_z], device=device, dtype=torch.float32)
        
        right = torch.cross(world_up, look)
        right = right / (right.norm() + 1e-8)

        up = torch.cross(look, right)
        up = up / (up.norm() + 1e-8)

        return look, right, up

    def build_camera_RT(self, az, el, dist, device):
        look, right, up = self.get_camera_vectors(az, el, device)

        eye = -look * dist

        # Camera basis as matrix rows:
        #  -right : flips the horizontal axis so the view isn't left-right mirrored
        #  -up    : keeps the image upright (nvdiffrast bottom-left framebuffer origin)
        #  -look  : forward axis
        R = torch.stack([-right, -up, -look], dim=0)

        # Translation
        T = -R @ eye

        return R, T

    def ortho_projection(self, scale, device):
        l = -scale * 0.5
        r =  scale * 0.5
        b = -scale * 0.5
        t =  scale * 0.5
        n = 0.1
        f = 100.0

        return torch.tensor([
            [2/(r-l),     0,          0,        -(r+l)/(r-l)],
            [0,           2/(t-b),    0,        -(t+b)/(t-b)],
            [0,           0,         -2/(f-n),  -(f+n)/(f-n)],
            [0,           0,          0,         1]
        ], dtype=torch.float32, device=device)

    # Check the existing PBR base color texture
    def getBaseColorTexture(self, mesh):
        if not hasattr(mesh.visual, "material"):
            mesh.visual.to_texture()
        if hasattr(mesh.visual, "material"):
            mat = mesh.visual.material
            existing_base = getattr(mat, 'baseColorTexture', None)
            if existing_base is None:
                # Fallback: try accessing via image attribute (SimpleMaterial / PBRMaterial variants)
                existing_base = getattr(mat, 'image', None)
        else:
            existing_base = None
        return existing_base
        
class Trellis2SelectImagesForMultiView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "front": ("STRING",{"default":""}),
                "preprocess": ("BOOLEAN",{"default":False}),
                "padding": ("INT",{"default":0,"min":0,"max":1024}),
                "remove_background": ("BOOLEAN",{"default":False}),
                "max_size": ("INT",{"default":2048,"min":512,"max":8192,"step":128}),
            },
            "optional":{
                "back": ("STRING",{"default":""}),
                "left": ("STRING",{"default":""}),
                "right": ("STRING",{"default":""})
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("front", "back", "left", "right",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def load_view(self, path):
        """Resolve a path (absolute, or relative to the ComfyUI input dir) to an IMAGE tensor."""
        if path is None or str(path).strip() == '':
            return None

        path = str(path).strip()
        if not os.path.exists(path):
            path = os.path.join(folder_paths.get_input_directory(), path)
            if not os.path.exists(path):
                return None

        image = Image.open(path)
        image = image.convert("RGBA" if 'A' in image.getbands() else "RGB")
        return pil2tensor(image)

    def process(self, front, preprocess, padding, remove_background, max_size, back = None, left = None, right = None):

        front_image = self.load_view(front)
        back_image = self.load_view(back)
        left_image = self.load_view(left)
        right_image = self.load_view(right)

        if front_image is None:
            raise ValueError(f"Trellis2SelectImagesForMultiView: could not find front image '{front}'")

        if preprocess:
            t2preprocess = Trellis2PreProcessImage()

            # process() is a ComfyUI node function: it returns a 1-tuple, so unwrap it.
            if front_image is not None:
                front_image = t2preprocess.process(front_image, padding, remove_background, max_size)[0]
            if back_image is not None:
                back_image = t2preprocess.process(back_image, padding, remove_background, max_size)[0]
            if left_image is not None:
                left_image = t2preprocess.process(left_image, padding, remove_background, max_size)[0]
            if right_image is not None:
                right_image = t2preprocess.process(right_image, padding, remove_background, max_size)[0]

        return (front_image, back_image, left_image, right_image, )
        
        
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
    "Trellis2MeshTexturing": Trellis2MeshTexturing,
    "Trellis2LoadMesh": Trellis2LoadMesh,
    "Trellis2PreProcessImage": Trellis2PreProcessImage,
    "Trellis2MeshRefiner": Trellis2MeshRefiner,
    "Trellis2PostProcess2": Trellis2PostProcess2,
    "Trellis2OvoxelExportToGLB": Trellis2OvoxelExportToGLB,
    "Trellis2TrimeshToMeshWithVoxel": Trellis2TrimeshToMeshWithVoxel,
    "Trellis2SimplifyTrimesh": Trellis2SimplifyTrimesh,
    "Trellis2Continue": Trellis2Continue,
    "Trellis2ProgressiveSimplify": Trellis2ProgressiveSimplify,
    "Trellis2ReconstructMesh": Trellis2ReconstructMesh,
    "Trellis2MeshWithVoxelToMeshlibMesh": Trellis2MeshWithVoxelToMeshlibMesh,
    "Trellis2FillHolesWithMeshlib": Trellis2FillHolesWithMeshlib,
    "Trellis2SmoothNormals": Trellis2SmoothNormals,
    "Trellis2RemeshWithQuad": Trellis2RemeshWithQuad,
    "Trellis2BatchSimplifyMeshAndExport": Trellis2BatchSimplifyMeshAndExport,
    "Trellis2MeshWithVoxelMultiViewGenerator": Trellis2MeshWithVoxelMultiViewGenerator,
    "Trellis2MeshTexturingMultiView": Trellis2MeshTexturingMultiView,
    "Trellis2WeldVertices": Trellis2WeldVertices,
    "Trellis2ReconstructMeshWithQuad": Trellis2ReconstructMeshWithQuad,
    "Trellis2StringSelector": Trellis2StringSelector,
    "Trellis2FillHolesWithCuMesh": Trellis2FillHolesWithCuMesh,
    "Trellis2LaplacianSmoothingWithOpen3d": Trellis2LaplacianSmoothingWithOpen3d,
    "Trellis2UnWrapTrimesh": Trellis2UnWrapTrimesh,
    "Trellis2MeshWithVoxelCascadeGenerator": Trellis2MeshWithVoxelCascadeGenerator,
    "Trellis2ImageCondGenerator": Trellis2ImageCondGenerator,
    "Trellis2SparseGenerator": Trellis2SparseGenerator,
    "Trellis2ShapeGenerator": Trellis2ShapeGenerator,
    "Trellis2ShapeCascadeGenerator": Trellis2ShapeCascadeGenerator,
    "Trellis2TexSlatGenerator": Trellis2TexSlatGenerator,
    "Trellis2DecodeLatents": Trellis2DecodeLatents,
    "Trellis2SimplifyMeshAdvanced": Trellis2SimplifyMeshAdvanced,
    "Trellis2SimplifyTrimeshAdvanced": Trellis2SimplifyTrimeshAdvanced,
    "Trellis2MultiViewTexturing": Trellis2MultiViewTexturing,
    "Trellis2Continue3": Trellis2Continue3,
    "Trellis2Continue4": Trellis2Continue4,
    "Trellis2Continue5": Trellis2Continue5,
    "Trellis2Continue6": Trellis2Continue6,
    "Trellis2CudaReset": Trellis2CudaReset,
    "Trellis2ProjectHighPolyToLowPoly": Trellis2ProjectHighPolyToLowPoly,
    "Trellis2RenderMultiView": Trellis2RenderMultiView,
    "Trellis2SaveImage": Trellis2SaveImage,
    "Trellis2VoxelToMesh": Trellis2VoxelToMesh,
    "Trellis2UnloadAllModels": Trellis2UnloadAllModels,
    "Trellis2SparseGeneratorWithReconViaGen": Trellis2SparseGeneratorWithReconViaGen,
    "Trellis2ExtractImagesFromVideo": Trellis2ExtractImagesFromVideo,
    "Trellis2MaxTokensCalculator": Trellis2MaxTokensCalculator,
    "Trellis2FillHolesNicelyWithMeshlib": Trellis2FillHolesNicelyWithMeshlib,
    "Trellis2SparseMultiViewGenerator": Trellis2SparseMultiViewGenerator,
    "Trellis2ImageCondMultiViewGenerator": Trellis2ImageCondMultiViewGenerator,
    "Trellis2ShapeMultiViewGenerator": Trellis2ShapeMultiViewGenerator,
    "Trellis2ShapeCascadeMultiViewGenerator": Trellis2ShapeCascadeMultiViewGenerator,
    "Trellis2TexSlatMultiViewGenerator": Trellis2TexSlatMultiViewGenerator,
    "Trellis2MoGeCameraConfig": Trellis2MoGeCameraConfig,
    "Trellis2FovMoGeCameraConfig": Trellis2FovMoGeCameraConfig,
    "Trellis2ImagesToViewConfigs": Trellis2ImagesToViewConfigs,
    "Trellis2ViewConfigsToList": Trellis2ViewConfigsToList,
    "Trellis2ExtractViewConfigDetails": Trellis2ExtractViewConfigDetails,
    "Trellis2LoadImagesFromFolder": Trellis2LoadImagesFromFolder,
    "Trellis2RenderMultiViewNvdiffrast": Trellis2RenderMultiViewNvdiffrast,
    "Trellis2SelectImagesForMultiView": Trellis2SelectImagesForMultiView,
    "Trellis2SmoothMeshWithPyMeshlab": Trellis2SmoothMeshWithPyMeshlab,
    "Trellis2SmoothTrimeshWithPyMeshlab": Trellis2SmoothTrimeshWithPyMeshlab,
    }
    

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2LoadModel": "Trellis2 - LoadModel",
    "Trellis2MeshWithVoxelGenerator": "Trellis2 - Mesh With Voxel Generator",
    "Trellis2LoadImageWithTransparency": "Trellis2 - Load Image with Transparency",
    "Trellis2SimplifyMesh": "Trellis2 - Simplify Mesh",
    "Trellis2MeshWithVoxelToTrimesh": "Trellis2 - Mesh With Voxel To Trimesh",
    "Trellis2ExportMesh": "Trellis2 - Export Mesh",
    "Trellis2PostProcessMesh": "Trellis2 - PostProcess Mesh (using Cumesh)",
    "Trellis2UnWrapAndRasterizer": "Trellis2 - UV Unwrap and Rasterize",
    "Trellis2MeshWithVoxelAdvancedGenerator": "Trellis2 - Mesh With Voxel Advanced Generator",
    "Trellis2PostProcessAndUnWrapAndRasterizer": "Trellis2 - Post Process/UnWrap and Rasterize",
    "Trellis2Remesh": "Trellis2 - Remesh",
    "Trellis2MeshTexturing": "Trellis2 - Mesh Texturing",
    "Trellis2LoadMesh": "Trellis2 - Load Mesh",
    "Trellis2PreProcessImage": "Trellis2 - PreProcess Image",
    "Trellis2MeshRefiner": "Trellis2 - Mesh Refiner",
    "Trellis2PostProcess2": "Trellis2 - PostProcess Mesh (using Trimesh)",
    "Trellis2OvoxelExportToGLB": "Trellis2 - Ovoxel Export to GLB",
    "Trellis2TrimeshToMeshWithVoxel": "Trellis2 - Trimesh to Mesh with Voxel",
    "Trellis2SimplifyTrimesh": "Trellis2 - Simplify Trimesh",
    "Trellis2Continue": "Trellis2 - Continue",
    "Trellis2ProgressiveSimplify": "Trellis2 - Progressive Simplify",
    "Trellis2ReconstructMesh": "Trellis2 - Reconstruct Mesh",
    "Trellis2MeshWithVoxelToMeshlibMesh": "Trellis2 - Mesh with Voxel to Meshlib Mesh",
    "Trellis2FillHolesWithMeshlib": "Trellis2 - Fill Holes with Meshlib",
    "Trellis2SmoothNormals": "Trellis2 - Smooth Normals",
    "Trellis2RemeshWithQuad": "Trellis2 - Remesh With Quad",
    "Trellis2BatchSimplifyMeshAndExport": "Trellis2 - Batch Simplify Mesh And Export",
    "Trellis2MeshWithVoxelMultiViewGenerator": "Trellis2 - Mesh With Voxel Multi-View Generator",
    "Trellis2MeshTexturingMultiView": "Trellis2 - Mesh Texturing Multi-View",
    "Trellis2WeldVertices": "Trellis2 - Weld Vertices",
    "Trellis2ReconstructMeshWithQuad": "Trellis2 - Reconstruct Mesh With Quad",
    "Trellis2StringSelector": "Trellis2 - String Selector",
    "Trellis2FillHolesWithCuMesh": "Trellis2 - Fill Holes with CuMesh",
    "Trellis2LaplacianSmoothingWithOpen3d": "Trellis2 - Laplacian Smoothing (using open3d)",
    "Trellis2UnWrapTrimesh": "Trellis2 - UnWrap Trimesh",
    "Trellis2MeshWithVoxelCascadeGenerator": "Trellis2 - Mesh With Voxel Cascade Generator",
    "Trellis2ImageCondGenerator": "Trellis2 - ImageCond Generator",
    "Trellis2SparseGenerator": "Trellis2 - Sparse Generator",
    "Trellis2ShapeGenerator": "Trellis2 - Shape Generator",
    "Trellis2ShapeCascadeGenerator": "Trellis2 - Shape Cascade Generator",
    "Trellis2TexSlatGenerator": "Trellis2 - Tex Slat Generator",
    "Trellis2DecodeLatents": "Trellis2 - Decode Latents",
    "Trellis2SimplifyMeshAdvanced": "Trellis2 - Simplify Mesh Advanced",
    "Trellis2SimplifyTrimeshAdvanced": "Trellis2 - Simplify Trimesh Advanced",
    "Trellis2MultiViewTexturing": "Trellis2 - Projection MultiView Texturing",
    "Trellis2Continue3": "Trellis2 - Continue 3",
    "Trellis2Continue4": "Trellis2 - Continue 4",
    "Trellis2Continue5": "Trellis2 - Continue 5",
    "Trellis2Continue6": "Trellis2 - Continue 6",
    "Trellis2CudaReset": "Trellis2 - Cuda Reset",
    "Trellis2ProjectHighPolyToLowPoly": "Trellis2 - Projection HighPoly To LowPoly",
    "Trellis2RenderMultiView": "Trellis2 - Render MultiView (Blender)",
    "Trellis2SaveImage": "Trellis2 - Save Image",
    "Trellis2VoxelToMesh": "Trellis2 - Voxel to Mesh",
    "Trellis2UnloadAllModels": "Trellis2 - Unload All ComfyUI Models",
    "Trellis2SparseGeneratorWithReconViaGen": "Trellis2 - Sparse Generator with ReconViaGen",
    "Trellis2ExtractImagesFromVideo": "Trellis2 - Extract Images from Video",
    "Trellis2MaxTokensCalculator": "Trellis2 - Max Tokens Calculator",
    "Trellis2FillHolesNicelyWithMeshlib": "Trellis2 - Fill Holes Nicely With Meshlib",
    "Trellis2SparseMultiViewGenerator": "Trellis2 - Sparse MultiView Generator",
    "Trellis2ImageCondMultiViewGenerator": "Trellis2 - ImageCond MultiView Generator",
    "Trellis2ShapeMultiViewGenerator": "Trellis2 - Shape MultiView Generator",
    "Trellis2ShapeCascadeMultiViewGenerator": "Trellis2 - Shape Cascade MultiView Generator",
    "Trellis2TexSlatMultiViewGenerator": "Trellis2 - Tex Slat MultiView Generator",
    "Trellis2MoGeCameraConfig": "Trellis2 - MoGe Camera Config",
    "Trellis2FovMoGeCameraConfig": "Trellis2 - Fov MoGe Camera Config",
    "Trellis2ImagesToViewConfigs": "Trellis2 - Images To ViewConfigs",
    "Trellis2ViewConfigsToList": "Trellis2 - ViewConfigs To List",
    "Trellis2ExtractViewConfigDetails": "Trellis2 - Extract ViewConfig Details",
    "Trellis2LoadImagesFromFolder": "Trellis2 - Load Images From Folder",
    "Trellis2RenderMultiViewNvdiffrast": "Trellis2 - Render MultiView (Nvdiffrast)",
    "Trellis2SelectImagesForMultiView": "Trellis2 - Select Images For MultiView",
    "Trellis2SmoothMeshWithPyMeshlab": "Trellis2 - Smooth Mesh With PyMeshlab",
    "Trellis2SmoothTrimeshWithPyMeshlab": "Trellis2 - Smooth Trimesh With PyMeshlab",
    }
