from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel

from .. import models

import gc
import os
import folder_paths
import trimesh
import o_voxel
import cumesh
import nvdiffrast.torch as dr
import cv2
import flex_gemm


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    # model_names_to_load = [
        # 'sparse_structure_flow_model',
        # 'sparse_structure_decoder',
        # 'shape_slat_flow_model_512',
        # 'shape_slat_flow_model_1024',
        # 'shape_slat_decoder',
        # 'tex_slat_flow_model_512',
        # 'tex_slat_flow_model_1024',
        # 'tex_slat_decoder',
    # ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json", keep_models_loaded = True) -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        #pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        #pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.image_cond_model = None
        pipeline.rembg_model = None
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'
        pipeline.path = path
        pipeline.keep_models_loaded = keep_models_loaded
        pipeline.last_processing = ''
        
        pipeline._pretrained_args['models']['sparse_structure_decoder'] = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS-image-large","ckpts","ss_dec_conv3d_16l8_fp16")
        facebook_model_path = os.path.join(folder_paths.models_dir,"facebook","dinov3-vitl16-pretrain-lvd1689m")
        pipeline._pretrained_args['image_cond_model']['args']['model_name'] = facebook_model_path           

        return pipeline
        
    def load_sparse_structure_model(self):        
        if self.models['sparse_structure_flow_model'] is None:
            print('Loading Sparse Structure model ...')
            self.models['sparse_structure_flow_model'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['sparse_structure_flow_model']}")
            self.models['sparse_structure_flow_model'].eval()
            self.models['sparse_structure_flow_model'].to(self._device)
        
        if self.models['sparse_structure_decoder'] is None:            
            self.models['sparse_structure_decoder'] = models.from_pretrained(self._pretrained_args['models']['sparse_structure_decoder'])
            self.models['sparse_structure_flow_model'].eval()        
            self.models['sparse_structure_decoder'].to(self._device)
    
    def unload_sparse_structure_model(self):
        if self.models['sparse_structure_flow_model']:
            del self.models['sparse_structure_flow_model']
            self.models['sparse_structure_flow_model'] = None
            gc.collect()
            
        if self.models['sparse_structure_decoder']:
            del self.models['sparse_structure_decoder']
            self.models['sparse_structure_decoder'] = None
            gc.collect()         
            
    def load_image_cond_model(self):
        if self.image_cond_model is None:
            print('Loading Image Cond model ...')
            self.image_cond_model = getattr(image_feature_extractor, self._pretrained_args['image_cond_model']['name'])(**self._pretrained_args['image_cond_model']['args'])
            self.image_cond_model.to(self._device)
            
    def unload_image_cond_model(self):
        if self.image_cond_model is not None:
            del self.image_cond_model
            self.image_cond_model = None
            gc.collect()
            
    def load_shape_slat_flow_model_512(self):        
        if self.models['shape_slat_flow_model_512'] is None:
            print('Loading Shape Slat Flow 512 model ...')
            self.models['shape_slat_flow_model_512'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['shape_slat_flow_model_512']}")
            self.models['shape_slat_flow_model_512'].eval()
            self.models['shape_slat_flow_model_512'].to(self._device)
            
    def unload_shape_slat_flow_model_512(self):
        if self.models['shape_slat_flow_model_512'] is not None:
            del self.models['shape_slat_flow_model_512']
            self.models['shape_slat_flow_model_512'] = None
            gc.collect()
            
    def load_tex_slat_flow_model_512(self):        
        if self.models['tex_slat_flow_model_512'] is None:
            print('Loading Texture Slat Flow 512 model ...')
            self.models['tex_slat_flow_model_512'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['tex_slat_flow_model_512']}")
            self.models['tex_slat_flow_model_512'].eval()
            self.models['tex_slat_flow_model_512'].to(self._device)

    def unload_tex_slat_flow_model_512(self):
        if self.models['tex_slat_flow_model_512'] is not None:
            del self.models['tex_slat_flow_model_512']
            self.models['tex_slat_flow_model_512'] = None
            gc.collect() 

    def load_tex_slat_decoder(self):        
        if self.models['tex_slat_decoder'] is None:
            print('Loading Texture Slat decoder model ...')
            self.models['tex_slat_decoder'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['tex_slat_decoder']}")
            self.models['tex_slat_decoder'].eval()
            self.models['tex_slat_decoder'].to(self._device)

    def unload_tex_slat_decoder(self):
        if self.models['tex_slat_decoder'] is not None:
            del self.models['tex_slat_decoder']
            self.models['tex_slat_decoder'] = None
            gc.collect()
            
    def load_shape_slat_decoder(self):        
        if self.models['shape_slat_decoder'] is None:
            print('Loading Shape Slat decoder model ...')
            self.models['shape_slat_decoder'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['shape_slat_decoder']}")
            self.models['shape_slat_decoder'].eval()
            self.models['shape_slat_decoder'].to(self._device)

    def unload_shape_slat_decoder(self):
        if self.models['shape_slat_decoder'] is not None:
            del self.models['shape_slat_decoder']
            self.models['shape_slat_decoder'] = None
            gc.collect()         

    def load_shape_slat_flow_model_1024(self):        
        if self.models['shape_slat_flow_model_1024'] is None:
            print('Loading Shape Slat Flow 1024 model ...')
            self.models['shape_slat_flow_model_1024'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['shape_slat_flow_model_1024']}")
            self.models['shape_slat_flow_model_1024'].eval()
            self.models['shape_slat_flow_model_1024'].to(self._device)

    def unload_shape_slat_flow_model_1024(self):
        if self.models['shape_slat_flow_model_1024'] is not None:
            del self.models['shape_slat_flow_model_1024']
            self.models['shape_slat_flow_model_1024'] = None
            gc.collect()   

    def load_tex_slat_flow_model_1024(self):        
        if self.models['tex_slat_flow_model_1024'] is None:
            print('Loading Texture Slat Flow 1024 model ...')
            self.models['tex_slat_flow_model_1024'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['tex_slat_flow_model_1024']}")
            self.models['tex_slat_flow_model_1024'].eval()
            self.models['tex_slat_flow_model_1024'].to(self._device)

    def unload_tex_slat_flow_model_1024(self):
        if self.models['tex_slat_flow_model_1024'] is not None:
            del self.models['tex_slat_flow_model_1024']
            self.models['tex_slat_flow_model_1024'] = None
            gc.collect()      

    def load_shape_slat_encoder(self):        
        if self.models['shape_slat_encoder'] is None:
            print('Loading Shape Slat Encoder model ...')
            self.models['shape_slat_encoder'] = models.from_pretrained(f"{self.path}/ckpts/shape_enc_next_dc_f16c32_fp16")
            self.models['shape_slat_encoder'].eval()
            self.models['shape_slat_encoder'].to(self._device)

    def unload_shape_slat_encoder(self):
        if self.models['shape_slat_encoder'] is not None:
            del self.models['shape_slat_encoder']
            self.models['shape_slat_encoder'] = None
            gc.collect()               

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            if self.image_cond_model is not None:
                self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
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
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
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
        
    def get_cond(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        resolution: int,
        include_neg_cond: bool = True,
        *,
        fusion_mode: str = "concat",   # "concat" or "mean"
        max_views: int = 4,            # safety cap for 3090
    ) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image: One of:
                - PIL Image
                - list[PIL Image] (multi-view)
                - torch.Tensor batch (B,H,W,C) from ComfyUI
            resolution: Conditioning resolution (e.g. 512 or 1024)
            include_neg_cond: Whether to include negative conditioning
            fusion_mode: "concat" (recommended) or "mean"
            max_views: Max number of views to fuse when list/batch is provided

        Returns:
            dict with keys: cond (+ neg_cond if include_neg_cond)
        """
        self.image_cond_model.image_size = resolution

        # ---- Normalize input into what image_cond_model expects ----
        # Most implementations expect PIL image or list[PIL images].
        if isinstance(image, torch.Tensor):
            # Expect ComfyUI IMAGE tensor: (B,H,W,C) float in [0,1]
            if image.ndim == 4:
                # Lazy import to avoid circulars if tensor2pil is in nodes/utils
                from .nodes import tensor2pil 
                images = [tensor2pil(image[i]) for i in range(min(int(image.shape[0]), max_views))]
            else:
                raise ValueError(f"Expected image tensor with shape (B,H,W,C), got {tuple(image.shape)}")
        elif isinstance(image, Image.Image):
            images = [image]
        elif isinstance(image, (list, tuple)):
            # list of PIL images
            images = list(image)[:max_views]
            if not images:
                raise ValueError("Empty image list provided to get_cond().")
            if not all(isinstance(im, Image.Image) for im in images):
                raise TypeError("get_cond() received a list/tuple but not all elements are PIL Images.")
        else:
            raise TypeError(f"Unsupported image type for get_cond(): {type(image)}")

        if self.low_vram:
            self.image_cond_model.to(self.device)

        # ---- Extract per-view conditioning ----
        cond = self.image_cond_model(images)

        # Normalize shapes:
        # Common outputs:
        #   - (V, N, D) for multi-view
        #   - (1, N, D) for single-view list length=1
        #   - (N, D) for some single-image extractors
        if cond.ndim == 2:
            # (N, D) -> (1, N, D)
            cond = cond.unsqueeze(0)
        elif cond.ndim != 3:
            raise RuntimeError(f"Unexpected cond ndim={cond.ndim}, shape={tuple(cond.shape)}")

        # If we passed multiple views, fuse them into one conditioning sequence
        if cond.shape[0] > 1:
            if fusion_mode == "concat":
                # (V, N, D) -> (1, V*N, D)
                cond = cond.reshape(1, -1, cond.shape[-1])
            elif fusion_mode == "mean":
                # (V, N, D) -> (1, N, D)
                cond = cond.mean(dim=0, keepdim=True)
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        if self.low_vram:
            self.image_cond_model.cpu()

        if not include_neg_cond:
            return {"cond": cond}

        neg_cond = torch.zeros_like(cond)
        return {"cond": cond, "neg_cond": neg_cond}

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample        
        self.load_shape_slat_decoder()
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        
        if not self.keep_models_loaded:
            self.unload_shape_slat_decoder()
        
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        
        self.load_shape_slat_decoder()
        
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        
        if not self.keep_models_loaded:        
            self.unload_shape_slat_decoder()
            
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor] = None,
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        
        self.load_tex_slat_decoder()
        
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
            
        if subs is None:
            ret = self.models['tex_slat_decoder'](slat) * 0.5 + 0.5
        else:
            ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
            
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        
        if not self.keep_models_loaded:
            self.unload_tex_slat_decoder()
        
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        sparse_structure_resolution: int = 32,
        max_views: int = 4        
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        # if pipeline_type == '512':
            # assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        # elif pipeline_type == '1024':
            # assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        # elif pipeline_type == '1024_cascade':
            # assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            # assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        # elif pipeline_type == '1536_cascade':
            # assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            # assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        # else:
            # raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        # Accept either a single PIL image or a list of PIL images (multi-view)
        if isinstance(image, (list, tuple)):
            images = list(image)
        else:
            images = [image]

        if preprocess_image:
            images = [self.preprocess_image(im) for im in images]
            
        torch.manual_seed(seed)
        
        # Get Image Cond
        self.load_image_cond_model()        
        # Multi-view conditioning happens inside get_cond()        
        cond_512  = self.get_cond(images, 512, max_views = max_views)
        cond_1024 = self.get_cond(images, 1024, max_views = max_views) if pipeline_type != '512' else None     
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()
        
        #ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        
        # Sampling Sparse Structure
        self.load_sparse_structure_model()        
        coords = self.sample_sparse_structure(
            cond_512, sparse_structure_resolution,
            num_samples, sparse_structure_sampler_params
        )
        
        if not self.keep_models_loaded:
            self.unload_sparse_structure_model()
        
        # Sampling Shape
        if pipeline_type == '512':            
            self.unload_shape_slat_flow_model_1024()
            self.load_shape_slat_flow_model_512()            
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
            
            self.unload_tex_slat_flow_model_1024()
            self.load_tex_slat_flow_model_512()
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_512()
            
            res = 512
        elif pipeline_type == '1024':
            self.unload_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_1024()
            
            self.unload_tex_slat_flow_model_512()
            self.load_tex_slat_flow_model_1024()
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
                
            res = 1024
        elif pipeline_type == '1024_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()            
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
            
            self.unload_tex_slat_flow_model_512()
            self.load_tex_slat_flow_model_1024()
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
        elif pipeline_type == '1536_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()                
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
            
            self.unload_tex_slat_flow_model_512()
            self.load_tex_slat_flow_model_1024()
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
            
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        torch.cuda.empty_cache()
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh

    def preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Preprocess the input mesh.
        """
        vertices = mesh.vertices
        vertices_min = vertices.min(axis=0)
        vertices_max = vertices.max(axis=0)
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale
        tmp = vertices[:, 1].copy()
        vertices[:, 1] = -vertices[:, 2]
        vertices[:, 2] = tmp
        assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), 'vertices out of range'
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)

    def encode_shape_slat(
        self,
        mesh: trimesh.Trimesh,
        resolution: int = 1024,
    ) -> SparseTensor:
        """
        Encode the meshes to structured latent.

        Args:
            mesh (trimesh.Trimesh): The mesh to encode.
            resolution (int): The resolution of mesh
        
        Returns:
            SparseTensor: The encoded structured latent.
        """
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
            
        vertices = SparseTensor(
            feats=dual_vertices * resolution - voxel_indices,
            coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)
        ).to(self.device)
        intersected = vertices.replace(intersected).to(self.device)
            
        self.load_shape_slat_encoder()
            
        if self.low_vram:
            self.models['shape_slat_encoder'].to(self.device)
        shape_slat = self.models['shape_slat_encoder'](vertices, intersected)
        if self.low_vram:
            self.models['shape_slat_encoder'].cpu()
            
        if not self.keep_models_loaded:
            self.unload_shape_slat_encoder()
            
        return shape_slat

    def postprocess_mesh(
        self,
        mesh: trimesh.Trimesh,
        pbr_voxel: SparseTensor,
        resolution: int = 1024,
        texture_size: int = 1024,
        texture_alpha_mode = 'OPAQUE',
        double_side_material = True
    ):
        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.vertex_normals
        vertices_torch = torch.from_numpy(vertices).float().cuda()
        faces_torch = torch.from_numpy(faces).int().cuda()
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = mesh.visual.uv.copy()
            uvs[:, 1] = 1 - uvs[:, 1]
            uvs_torch = torch.from_numpy(uvs).float().cuda()
        else:
            _cumesh = cumesh.CuMesh()
            _cumesh.init(vertices_torch, faces_torch)
            print('Unwrapping mesh ...')
            vertices_torch, faces_torch, uvs_torch, vmap = _cumesh.uv_unwrap(return_vmaps=True)
            vertices_torch = vertices_torch.cuda()
            faces_torch = faces_torch.cuda()
            uvs_torch = uvs_torch.cuda()
            vertices = vertices_torch.cpu().numpy()
            faces = faces_torch.cpu().numpy()
            uvs = uvs_torch.cpu().numpy()
            normals = normals[vmap.cpu().numpy()]
                
        # rasterize
        print('Finalizing mesh ...')
        ctx = dr.RasterizeCudaContext()
        uvs_torch = torch.cat([uvs_torch * 2 - 1, torch.zeros_like(uvs_torch[:, :1]), torch.ones_like(uvs_torch[:, :1])], dim=-1).unsqueeze(0)
        rast, _ = dr.rasterize(
            ctx, uvs_torch, faces_torch,
            resolution=[texture_size, texture_size],
        )
        mask = rast[0, ..., 3] > 0
        pos = dr.interpolate(vertices_torch.unsqueeze(0), rast, faces_torch)[0][0]
        
        attrs = torch.zeros(texture_size, texture_size, pbr_voxel.shape[1], device=self.device)
        attrs[mask] = flex_gemm.ops.grid_sample.grid_sample_3d(
            pbr_voxel.feats,
            pbr_voxel.coords,
            shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
            grid=((pos[mask] + 0.5) * resolution).reshape(1, -1, 3),
            mode='trilinear',
        )
        
        # construct mesh
        mask = mask.cpu().numpy()
        base_color = np.clip(attrs[..., self.pbr_attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., self.pbr_attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., self.pbr_attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., self.pbr_attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        
        # extend
        mask = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask, 1, cv2.INPAINT_TELEA)[..., None]
        
        baseColorTexture = Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
        metallicRoughnessTexture = Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1))
        
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=baseColorTexture,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=metallicRoughnessTexture,
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=texture_alpha_mode,
            doubleSided=True,
        )

        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
        normals[:, 1], normals[:, 2] = normals[:, 2], -normals[:, 1]
        uvs[:, 1] = 1 - uvs[:, 1] # Flip UV V-coordinate
        
        textured_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs, material=material)
        )
        
        return textured_mesh, baseColorTexture, metallicRoughnessTexture

    @torch.no_grad()
    def texture_mesh(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        resolution: int = 1024,
        texture_size: int = 2048,
        texture_alpha_mode = 'OPAQUE',
        double_side_material = True
    ):
        mesh = self.preprocess_mesh(mesh)
        torch.manual_seed(seed)
        
        self.load_image_cond_model()        
        cond = self.get_cond(image, resolution)
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()
        
        shape_slat = self.encode_shape_slat(mesh, resolution)
        
        if resolution==512:
            self.unload_tex_slat_flow_model_1024()
            self.load_tex_slat_flow_model_512()
            tex_model = self.models['tex_slat_flow_model_512']
            
            tex_slat = self.sample_tex_slat(
                cond, tex_model,
                shape_slat, tex_slat_sampler_params
            )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_512()
        else:
            self.unload_tex_slat_flow_model_512()
            self.load_tex_slat_flow_model_1024()
            tex_model = self.models['tex_slat_flow_model_1024']
            
            tex_slat = self.sample_tex_slat(
                cond, tex_model,
                shape_slat, tex_slat_sampler_params
            )
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_1024()

        torch.cuda.empty_cache()
        pbr_voxel = self.decode_tex_slat(tex_slat)
        torch.cuda.empty_cache()
        
        out_mesh, baseColorTexture, metallicRoughnessTexture = self.postprocess_mesh(mesh, pbr_voxel, resolution, texture_size, texture_alpha_mode, double_side_material)
        return out_mesh, baseColorTexture, metallicRoughnessTexture