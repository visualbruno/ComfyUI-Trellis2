"""
Multi-View Texture Projection for Trellis2

Pipeline:
  A) UV-space rasterization (once, shared across all views):
     - Rasterize mesh using UV coords as clip positions
     - For every texel: interpolate world-space position, normal, face_id

  B) Per-view projection (per view):
     - Rasterize mesh from camera -> face_id per camera pixel (occlusion)
     - Detect character pixel bounds in source image (background removal)
     - Map mesh world-space bounds -> image pixel bounds
     - For each UV texel:
         1. Project 3D position to camera space (world units)
         2. Map world coords -> image pixel coords
         3. Occlusion: face_id match (exact, integer comparison)
         4. Sample colour from source image
         5. Accumulate colour * weight (dot(normal, -look)^exp)

  C) Normalize + inpaint holes

Coordinate system (Y-up world):
  +X = character's LEFT  (from viewer's perspective looking at front)
  +Y = up
  +Z = character's front (toward viewer at az=0)

  Azimuth  0   -> camera at +Z, looking toward -Z (front view)
  Azimuth  90  -> camera at +X, looking toward -X
  Azimuth  180 -> camera at -Z, looking toward +Z (back view)
  Elevation 0  -> horizontal
  Elevation 90 -> camera above, looking down

Camera axes (derived):
  look  = normalize(origin - cam_pos)  (toward origin)
  right = normalize(look x world_up)   -- points to IMAGE right
  up    = normalize(right x look)      -- points to IMAGE up

  At az=0: cam_pos=(0,0,1), look=(0,0,-1)
    right = cross((0,0,-1), (0,1,0)) = (0*0-(-1)*1, (-1)*0-0*0, 0*1-0*0) = (1,0,0)  -> +X
    This means image-right = world +X = character's LEFT from viewer.
    The source images are rendered from the same camera convention, so this is consistent.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import nvdiffrast.torch as dr
import cv2
import trimesh
import gc
import math

import cumesh as CuMesh

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def get_camera_vectors(azimuth: float, elevation: float, device='cuda'):
    az = math.radians(azimuth)
    el = math.radians(elevation)

    cx = math.sin(az) * math.cos(el)
    cy = math.sin(el)
    cz = math.cos(az) * math.cos(el)

    look = torch.tensor([-cx, -cy, -cz], device=device, dtype=torch.float32)
    look = look / (look.norm() + 1e-8)
    world_up = torch.tensor([0., 1., 0.], device=device, dtype=torch.float32)

    right = torch.cross(look, world_up, dim=-1)
    if right.norm() < 1e-6:
        if look[1] > 0.0:  
            world_fwd = torch.tensor([0., 0., 1.], device=device, dtype=torch.float32)
        else:              
            world_fwd = torch.tensor([0., 0., -1.], device=device, dtype=torch.float32)
        right = torch.cross(look, world_fwd, dim=-1)        
    right = right / (right.norm() + 1e-8)

    up = torch.cross(right, look, dim=-1)
    up = up / (up.norm() + 1e-8)

    return look, right, up

def build_ortho_clip_verts(vertices, right, up, look, ortho_scale):
    """
    Matches Blender's Orthographic Camera.
    ortho_scale = 2.0 means camera views from -1.0 to +1.0 in world units.
    """
    s = max(float(ortho_scale), 1e-6)
    x = (vertices * right).sum(-1) / (s / 2.0)
    y = (vertices * up).sum(-1)    / (s / 2.0)
    d = -(vertices * look).sum(-1)
    
    # Standard depth remapping for nvdiffrast
    d_min  = d.min()
    d_span = (d.max() - d_min).clamp(min=1e-6)
    z      = 1.0 - 2.0 * (d - d_min) / d_span
    return torch.stack([x, y, z, torch.ones_like(x)], dim=-1).unsqueeze(0)

def project_texels_to_image(tex_pos, right, up, ortho_scale):
    """
    Converts 3D world positions to grid_sample compatible coordinates.
    """
    s = max(float(ortho_scale), 1e-6)
    
    # Map world units to clip space [-1, 1]
    # In Blender, ortho_scale is the full width, so radius is s/2
    u_clip = (tex_pos * right).sum(-1) / (s / 2.0)
    v_clip = (tex_pos * up).sum(-1) / (s / 2.0)
    
    # grid_sample expects v = -1 at the TOP of the image.
    # Our 'up' vector gives positive values at the top, so we invert it for sampling.
    u_samp = u_clip
    v_samp = -v_clip 
    
    return u_samp, v_samp, u_clip, v_clip


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def texture_mesh_with_multiview(
    high_poly_mesh: trimesh.Trimesh,
    images:     list,
    azimuths:   list,
    elevations: list,
    view_weights: list = None,
    texture_size: int   = 4096,
    mesh_cluster_threshold_cone_half_angle_rad: float = 60.0,
    mesh_cluster_refine_iterations:  int   = 0,
    mesh_cluster_global_iterations:  int   = 1,
    mesh_cluster_smooth_strength:    int   = 1,
    blend_exponent: float = 2.0,
    ortho_scale:    float = 1.0,
    norm_size: float = 1.15,
    fill_holes: bool = False,
    blend_texture: bool = True,
    max_hole_size: int = 10,
    use_metallic: bool = True,
    depth_eps: float = 0.002,
    low_poly_mesh: trimesh.Trimesh = None,
    add_alpha_channel: bool = False
):
    if not (len(images) == len(azimuths) == len(elevations)):
        raise ValueError("images, azimuths, and elevations must have the same length")
        
    if not (len(images) == len(azimuths) == len(elevations)):
        raise ValueError("images, azimuths, and elevations must have the same length")

    num_views = len(images)

    if view_weights is None:
        view_weights = [1.0] * num_views

    if len(view_weights) != num_views:
        raise ValueError("view_weights must match number of images")        

    num_views = len(images)
    print(f"[MultiView] {num_views} views | texture={texture_size} | ortho_scale={ortho_scale}")

    if low_poly_mesh is not None:
        mesh = low_poly_mesh
    else:
        mesh = high_poly_mesh
        
    # =========================================================================
    # STEP 1 – UV unwrap
    # =========================================================================
    #vertices = torch.from_numpy(mesh.vertices).float().cuda()
    #faces = torch.from_numpy(mesh.faces).int().cuda()
    #vertices = mesh.vertices.cuda()
    #faces    = mesh.faces.cuda()

    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        print('Mesh has UV')
        out_verts = torch.from_numpy(mesh.vertices).float().cuda()
        out_faces = torch.from_numpy(mesh.faces).int().cuda()
        out_uvs = torch.from_numpy(mesh.visual.uv).float().cuda()
        
        #mesh.fix_normals()
        if not hasattr(mesh, 'vertex_normals') or len(mesh.vertex_normals) == 0:
            print('Generating Normals ...')
            # Force generation of face normals first (required for vertex normals)
            mesh.face_normals = mesh.generate_face_normals()
            # Generate vertex normals weighted by face area
            mesh.vertex_normals = mesh.generate_vertex_normals()
        
        out_normals = torch.from_numpy(mesh.vertex_normals).float().cuda()        
        #out_normals = out_normals / (out_normals.norm(dim=-1, keepdim=True) + 1e-8)
        
    else:
        print("[MultiView] UV unwrapping...")
        cumesh = CuMesh.CuMesh()
        cumesh.init(torch.from_numpy(mesh.vertices).float().cuda(), torch.from_numpy(mesh.faces).int().cuda())
        out_verts, out_faces, out_uvs, out_vmaps = cumesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": np.radians(mesh_cluster_threshold_cone_half_angle_rad),
                "refine_iterations": mesh_cluster_refine_iterations,
                "global_iterations": mesh_cluster_global_iterations,
                "smooth_strength":   mesh_cluster_smooth_strength,
            },
            return_vmaps=True,
            verbose=True,
        )
        out_verts  = out_verts.cuda()
        out_faces  = out_faces.cuda()
        out_uvs    = out_uvs.cuda()

        cumesh.compute_vertex_normals()
        out_normals = cumesh.read_vertex_normals()[out_vmaps.cuda()]
        out_normals = out_normals / (out_normals.norm(dim=-1, keepdim=True) + 1e-8)

        del cumesh; gc.collect(); torch.cuda.empty_cache()

    # =========================================================================
    # NEW: Replicate blender_render.py normalization
    # =========================================================================
    # 1. Center the geometry (Blender does this in auto_center_and_scale)
    bbox_min = out_verts.min(dim=0)[0]
    bbox_max = out_verts.max(dim=0)[0]
    center = (bbox_min + bbox_max) / 2.0
    out_verts = out_verts - center

    # 2. Scale to match Blender's norm_size
    # Blender calculates Max Radius from the new center
    current_max_radius = torch.sqrt((out_verts**2).sum(dim=-1).max())
    norm_size = 1.15  # This must match --norm_size in blender_render.py
    scale_user = current_max_radius * 2.0
    scale_factor = norm_size / scale_user

    out_verts = out_verts * scale_factor

    print(f"[MultiView] Mesh: {out_verts.shape[0]} verts, {out_faces.shape[0]} faces")
    print(f"  X:[{out_verts[:,0].min():.3f},{out_verts[:,0].max():.3f}]"
          f"  Y:[{out_verts[:,1].min():.3f},{out_verts[:,1].max():.3f}]"
          f"  Z:[{out_verts[:,2].min():.3f},{out_verts[:,2].max():.3f}]")

    # REPLICATE BLENDER NORMALIZATION
    bbox_min = out_verts.min(dim=0)[0]
    bbox_max = out_verts.max(dim=0)[0]
    center = (bbox_min + bbox_max) / 2.0
    out_verts = out_verts - center # Center at origin

    current_max_radius = torch.sqrt((out_verts**2).sum(dim=-1).max())
    scale_factor = norm_size / (current_max_radius * 2.0)
    out_verts = out_verts * scale_factor # Scale to 1.15 units
    out_normals = out_normals / (out_normals.norm(dim=-1, keepdim=True) + 1e-8)

    # --- STEP 2: UV RASTERIZATION ---
    ctx = dr.RasterizeCudaContext()
    uvs_clip = torch.cat([out_uvs * 2.0 - 1.0, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast, _ = dr.rasterize(ctx, uvs_clip, out_faces.int(), resolution=[texture_size, texture_size])
    uv_hit_mask = rast[0, :, :, 3] > 0
    
    tex_pos = dr.interpolate(out_verts.unsqueeze(0), rast, out_faces.int())[0][0]
    tex_normals = dr.interpolate(out_normals.unsqueeze(0), rast, out_faces.int())[0][0]
    tex_normals = tex_normals / (tex_normals.norm(dim=-1, keepdim=True) + 1e-8)
    tex_face_id = rast[0, :, :, 3].long() - 1

    # --- STEP 3: VIEW PROJECTION ---
    acc_color  = torch.zeros(texture_size, texture_size, 3, device='cuda')
    acc_weight = torch.zeros(texture_size, texture_size, device='cuda')

    for img, az, el, view_w in zip(images, azimuths, elevations, view_weights):
        img_np = np.array(img.convert('RGB')).astype(np.float32) / 255.0
        img_h, img_w = img_np.shape[:2]

        # --- Dilate foreground colors into background to prevent bilinear bleed ---
        img_rgba = img.convert('RGBA')
        img_np_rgba = np.array(img_rgba)

        # Build foreground mask
        if img.mode != 'RGBA' or img_np_rgba[:, :, 3].min() == 255:
            bg_color = img_np_rgba[0, 0, :3]
            color_diff = np.abs(img_np_rgba[:, :, :3].astype(int) - bg_color.astype(int)).sum(axis=-1)
            fg_mask = (color_diff > 10).astype(np.uint8)
        else:
            fg_mask = (img_np_rgba[:, :, 3] > 127).astype(np.uint8)

        # Dilate the foreground mask and inpaint the background band
        # Scale dilation with image resolution to prevent bilinear bleed
        base_res = 2048
        scale = max(img_h, img_w) / base_res
        dilate_px = int(max(5, 5 * scale))
        kernel = np.ones((3, 3), np.uint8)
        dilated_fg = cv2.dilate(fg_mask, kernel, iterations=dilate_px)
        band_to_fill = cv2.bitwise_and(dilated_fg, cv2.bitwise_not(fg_mask))

        # Inpaint the color channels into the dilated band
        img_rgb_u8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        if int(band_to_fill.sum()) > 0:
            for c in range(3):
                img_rgb_u8[..., c] = cv2.inpaint(img_rgb_u8[..., c], band_to_fill, 3, cv2.INPAINT_NS)
        img_np = img_rgb_u8.astype(np.float32) / 255.0

        img_t = torch.from_numpy(img_np).cuda().permute(2, 0, 1).unsqueeze(0).contiguous()

        look, right, up = get_camera_vectors(az, el, device='cuda')

        # Create a moderately high-resolution occlusion map.
        # Side views are especially sensitive to self-occlusion leaks where
        # background/body texels can slip through around limbs.
        occ_res = min(2048, img_h, img_w)
        cam_clip = build_ortho_clip_verts(out_verts, right, up, look, ortho_scale)
        
        cam_rast, _ = dr.rasterize(ctx, cam_clip, out_faces.int(), resolution=[occ_res, occ_res])
        cam_hit = cam_rast[0, :, :, 3] > 0
        cam_hit_img = cam_hit.float().unsqueeze(0).unsqueeze(0)

        # depth buffer
        cam_depth = dr.interpolate(
            (-(out_verts * look).sum(-1)).unsqueeze(0).unsqueeze(-1),
            cam_rast,
            out_faces.int()
        )[0][0]

        cam_depth = cam_depth.permute(2,0,1).unsqueeze(0)  # for grid_sample
        inf_depth = torch.full_like(cam_depth, torch.finfo(cam_depth.dtype).max)
        cam_depth_occ = torch.where(cam_hit.unsqueeze(0).unsqueeze(0), cam_depth, inf_depth)
        cam_depth_occ = -F.max_pool2d(-cam_depth_occ, kernel_size=1, stride=1, padding=0)

        # Map texels to camera clip space
        _, _, u_clip, v_clip = project_texels_to_image(tex_pos, right, up, ortho_scale)

        # Map mesh world-space bounds -> image pixel bounds
        cam_clip_verts = cam_clip[0]
        mesh_u_min = cam_clip_verts[:, 0].min()
        mesh_u_max = cam_clip_verts[:, 0].max()
        mesh_v_min = cam_clip_verts[:, 1].min()
        mesh_v_max = cam_clip_verts[:, 1].max()

        mesh_u_span = (mesh_u_max - mesh_u_min).clamp(min=1e-6)
        mesh_v_span = (mesh_v_max - mesh_v_min).clamp(min=1e-6)

        # Find character bounds in image using the foreground mask
        coords = np.argwhere(fg_mask)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
        else:
            y_min, x_min = 0, 0
            y_max, x_max = img_h - 1, img_w - 1

        u_norm = (u_clip - mesh_u_min) / mesh_u_span
        x_pixel = u_norm * float(x_max - x_min + 1) + float(x_min) - 0.5
        u_samp = ((x_pixel + 0.5) / float(img_w)) * 2.0 - 1.0

        v_norm = (v_clip - mesh_v_min) / mesh_v_span
        y_pixel = (1.0 - v_norm) * float(y_max - y_min + 1) + float(y_min) - 0.5
        v_samp = ((y_pixel + 0.5) / float(img_h)) * 2.0 - 1.0

        # Prevent grid_sample from touching the border texel
        eps = 1e-4
        u_samp = u_samp.clamp(-1 + eps, 1 - eps)
        v_samp = v_samp.clamp(-1 + eps, 1 - eps)

        # Occlusion check
        grid_occ = torch.stack([u_clip, v_clip], dim=-1).unsqueeze(0)
        sampled_hit = F.grid_sample(
            cam_hit_img,
            grid_occ,
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )[0, 0] > 0.10
        
        sampled_depth = F.grid_sample(
            cam_depth_occ,
            grid_occ,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )[0,0]

        tex_depth = -(tex_pos * look).sum(-1)

        depth_match = tex_depth >= sampled_depth - depth_eps
        
        # Visibility Mask
        in_bounds = (u_samp.abs() <= 1.0) & (v_samp.abs() <= 1.0)
        visible = uv_hit_mask & in_bounds & sampled_hit & depth_match

        # Weighting and Accumulation
        grid_col = torch.stack([u_samp, v_samp], dim=-1).unsqueeze(0)
        sampled_colors = F.grid_sample(img_t, grid_col, mode='bilinear', padding_mode='border', align_corners=False)[0].permute(1, 2, 0)
        
        dot = (tex_normals * (-look)).sum(-1).clamp(min=0.0) ** 1.5
        weights = (dot ** blend_exponent) * visible.float() * view_w

        acc_color += sampled_colors * weights.unsqueeze(-1)
        acc_weight += weights

    # =========================================================================
    # STEP 4 – Normalize and blend over existing PBR texture
    # =========================================================================
    print("[MultiView] Finalising texture...")

    total_w = acc_weight.sum().item()
    print(f"  Total accumulated weight: {total_w:.1f}")
    if total_w < 1.0:
        print("  WARNING: almost no weight accumulated – check camera setup!")

    # Mask for safe division (avoid div-by-zero)
    valid_mask = acc_weight > 1e-6
    projected_color = torch.zeros_like(acc_color)
    projected_color[valid_mask] = acc_color[valid_mask] / acc_weight[valid_mask].unsqueeze(-1)

    # Mask for compositing: only replace original texture where at least one view
    # had a confident projection (normal dot product weight above a real threshold).
    # This prevents near-black grazing-angle texels from overwriting the original.
    # Threshold is relative to the maximum accumulated weight in the texture,
    # so it adapts automatically regardless of blend_exponent or number of views.
    # ------------------------------------------------------------------
    # Compute confidence from accumulated weights
    # ------------------------------------------------------------------

    max_w = acc_weight.max().clamp(min=1e-6)

    # Normalize weights
    confidence = acc_weight / max_w

    # Smoothstep curve (removes harsh edges)
    confidence = confidence * confidence * (3.0 - 2.0 * confidence)

    # Optional: boost confident projections
    confidence = confidence ** 0.5

    confidence3 = confidence.unsqueeze(-1)

    #-- Load and resample the existing PBR base color texture ----------------
    existing_base = None
    try:
        mat = high_poly_mesh.visual.material
        existing_base = getattr(mat, 'baseColorTexture', None)
        if existing_base is None:
            # Fallback: try accessing via image attribute (SimpleMaterial / PBRMaterial variants)
            existing_base = getattr(mat, 'image', None)
    except AttributeError:
        pass

    if existing_base and blend_texture:
        print("  Blending projected views over existing PBR texture...")
        if existing_base.mode != 'RGBA':
            existing_base = existing_base.convert('RGBA')
        # Resize to target texture size if needed
        if existing_base.size != (texture_size, texture_size):
            print(f"  Resampling existing texture from {existing_base.size} -> ({texture_size},{texture_size})")
            existing_base = existing_base.resize((texture_size, texture_size), Image.LANCZOS)

        existing_np = np.array(existing_base).astype(np.float32) / 255.0   # (H, W, 4)
        existing_np = np.flip(existing_np, axis=0).copy()
        existing_rgb = torch.from_numpy(existing_np[..., :3]).cuda()        # (H, W, 3)
        existing_alpha = torch.from_numpy(existing_np[..., 3:4]).cuda()     # (H, W, 1)

        # Hard composite: where the projection covered a texel, use the projected
        # color fully. Where it didn't, keep the original texture untouched.
        # The normal-based weighting already handles per-view confidence during
        # accumulation, so no additional blending factor is needed here.
        # blended_rgb = (
            # projected_color * confidence3 +
            # existing_rgb * (1.0 - confidence3)
        # )
        
        # Hard composite: use projected colour only where:
        #   - the texel lies inside a UV island (uv_hit_mask)
        #   - the accumulated projection weight is above a tiny fraction of the max weight
        conf_threshold = 0.1
        hard_mask = uv_hit_mask & (confidence > conf_threshold)

        # Keep original texture wherever the mask is False
        blended_rgb = torch.where(hard_mask.unsqueeze(-1), projected_color, existing_rgb)

        composite_mask = hard_mask  # for the reporting print below        

        #composite_mask = confidence > 0.01

        color_np = (blended_rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # Alpha from projection confidence
        conf_np = confidence.cpu().numpy()        
        
        if add_alpha_channel:
            uv_mask_np = uv_hit_mask.cpu().numpy()
            alpha_np = np.zeros((texture_size, texture_size), dtype=np.uint8)
            alpha_np[uv_mask_np] = (conf_np[uv_mask_np] * 255).clip(0, 255).astype(np.uint8)
            alpha_np[conf_np <= 0.01] = 0
        else:            
            # Preserve the original alpha everywhere (mesh already has full coverage)
            alpha_np = (existing_alpha.squeeze(-1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        n_projected = int(composite_mask.sum().item())
        print(f"  Projected texels: {n_projected} / {texture_size*texture_size}"
              f"  ({100.0*n_projected/(texture_size*texture_size):.1f}%)")

    else:
        # No existing texture found – fall back to projection-only output
        print(
            "  WARNING: No existing PBR baseColorTexture found on mesh, "
            "outputting projection-only texture (holes will be transparent)."
        )

        # RGB from projection
        color_np = (projected_color.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Alpha from projection confidence
        conf_np = confidence.cpu().numpy()

        if add_alpha_channel:
            # Use UV mask for proper transparent background
            uv_mask_np = uv_hit_mask.cpu().numpy()

            alpha_np = np.zeros((texture_size, texture_size), dtype=np.uint8)

            alpha_np[uv_mask_np] = (conf_np[uv_mask_np] * 255).clip(0, 255).astype(np.uint8)
            alpha_np[conf_np <= 0.01] = 0
        else:
            # Slight threshold to remove noise
            alpha_mask = conf_np > 0.01

            alpha_np = (conf_np * 255).clip(0, 255).astype(np.uint8)
            alpha_np[~alpha_mask] = 0

    if fill_holes:
        print('Filling holes and padding UV seams ...')
        
        # Get numpy arrays
        uv_mask_np = uv_hit_mask.cpu().numpy().astype(np.uint8)
        valid_mask_np = valid_mask.cpu().numpy().astype(np.uint8)
        
        # 1. INTERNAL HOLES: Pixels inside the UV map that didn't get colored
        internal_hole_mask = cv2.bitwise_and(cv2.bitwise_not(valid_mask_np), uv_mask_np)
        
        if max_hole_size > 0:
            # Filter internal holes by size to avoid filling massive intentional gaps
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(internal_hole_mask, connectivity=8)
            areas = stats[:, cv2.CC_STAT_AREA]
            valid_labels_mask = (areas <= max_hole_size)
            valid_labels_mask[0] = False  # Ignore background
            filtered_internal_holes = valid_labels_mask[labels].astype(np.uint8)
        else:
            filtered_internal_holes = internal_hole_mask

        # 2. UV SEAM PADDING: A small band of pixels just outside the UV islands
        # We dilate the UV mask by a few pixels to create a "bleed" area
        pad_radius = 5  # You can increase this if the black lines persist at long distances
        kernel = np.ones((3, 3), np.uint8)
        dilated_uv = cv2.dilate(uv_mask_np, kernel, iterations=pad_radius)
        
        # The padding mask is the dilated area MINUS the original UV area
        seam_padding_mask = cv2.bitwise_and(dilated_uv, cv2.bitwise_not(uv_mask_np))

        # 3. COMBINE AND INPAINT
        # We want to inpaint both the internal dots and the external seams simultaneously
        final_inpaint_mask = cv2.bitwise_or(filtered_internal_holes, seam_padding_mask)

        n_holes = int(filtered_internal_holes.sum())
        n_pad   = int(seam_padding_mask.sum())
        print(f"  Inpainting {n_holes} internal texels and {n_pad} seam padding texels...")

        if (n_holes + n_pad) > 0:
            for c in range(3):
                color_np[..., c] = cv2.inpaint(color_np[..., c], final_inpaint_mask, 3, cv2.INPAINT_NS)
            alpha_np = cv2.inpaint(alpha_np, final_inpaint_mask, 3, cv2.INPAINT_NS)      

    # =========================================================================
    # STEP 5 – Build output textures and trimesh
    # =========================================================================
    baseColorTexture = Image.fromarray(np.dstack([color_np, alpha_np]))
    metallicRoughnessTexture = None
    
    if hasattr(high_poly_mesh, 'visual') and hasattr(high_poly_mesh.visual, 'material') and isinstance(high_poly_mesh.visual.material, trimesh.visual.material.PBRMaterial):
        if high_poly_mesh.visual.material.metallicRoughnessTexture:
            metallicRoughnessTexture = high_poly_mesh.visual.material.metallicRoughnessTexture
            
    if metallicRoughnessTexture is None or not use_metallic:
        mr_np = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        mr_np[..., 1] = 230
        metallicRoughnessTexture = Image.fromarray(mr_np)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=baseColorTexture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=metallicRoughnessTexture,
        metallicFactor=0.0,
        roughnessFactor=0.9,
        alphaMode='OPAQUE',
        doubleSided=True,
    )

    verts_np = out_verts.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np   = out_uvs.cpu().numpy()
    uvs_np[:, 1] = 1.0 - uvs_np[:, 1]   # flip V: GL bottom-up -> image top-down

    textured_mesh = trimesh.Trimesh(
        vertices=verts_np,
        faces=faces_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
    )

    return textured_mesh, baseColorTexture, metallicRoughnessTexture
