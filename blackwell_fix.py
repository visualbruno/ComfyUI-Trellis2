"""
blackwell_fix.py — TRELLIS.2 Mesh Extraction Fix for NVIDIA Blackwell GPUs

Drop-in workaround for the broken CuMesh remeshing pipeline on sm_120 GPUs
(RTX 5070, 5070 Ti, 5080, 5090). Replaces the CUDA-dependent to_glb() mesh
extraction with a voxel-based marching cubes approach that produces watertight,
3D-printable meshes using only CPU operations.

Usage:
    import blackwell_fix

    # Auto-detect Blackwell and apply all compatibility patches
    blackwell_fix.patch_all()

    # After running TRELLIS.2 inference:
    mesh = pipeline.run(image)[0]
    trimesh_mesh = blackwell_fix.voxel_to_mesh(mesh)
    trimesh_mesh.export("output.stl")

Requirements:
    numpy, scipy, scikit-image, trimesh
"""

import os
import sys
import gc
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Blackwell GPU detection
# ---------------------------------------------------------------------------

def is_blackwell_gpu(device: int = 0) -> bool:
    """Check if the current GPU is a Blackwell-architecture device (sm_120)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(device)
        return major >= 12
    except Exception:
        return False


def get_gpu_info(device: int = 0) -> dict:
    """Return GPU name and compute capability."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"name": "N/A", "compute_capability": (0, 0), "is_blackwell": False}
        props = torch.cuda.get_device_properties(device)
        cc = torch.cuda.get_device_capability(device)
        return {
            "name": props.name,
            "compute_capability": cc,
            "is_blackwell": cc[0] >= 12,
        }
    except Exception:
        return {"name": "N/A", "compute_capability": (0, 0), "is_blackwell": False}


# ---------------------------------------------------------------------------
# Compatibility patches — must be applied BEFORE importing TRELLIS.2
# ---------------------------------------------------------------------------

_patches_applied = False


def patch_all(force: bool = False, verbose: bool = True):
    """
    Apply all Blackwell compatibility patches.

    Must be called BEFORE importing trellis2 or o_voxel. Sets environment
    variables and monkey-patches CUDA capability detection so that spconv,
    cumm, and flex_gemm select sm_90 (Hopper) PTX kernels, which JIT-compile
    correctly on Blackwell hardware.

    Args:
        force: Apply patches even on non-Blackwell GPUs (for testing).
        verbose: Print status messages for each patch applied.
    """
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    import torch

    # Check if patches are needed
    major, minor = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    if major < 10 and not force:
        if verbose:
            print(f"[blackwell_fix] GPU CC {major}.{minor} — no patches needed")
        return

    if verbose:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
        print(f"[blackwell_fix] Detected {name} (CC {major}.{minor}) — applying patches")

    # ── Environment variables (must be set before TRELLIS imports) ──
    os.environ["ATTN_BACKEND"] = "sdpa"              # PyTorch native SDPA attention
    os.environ["SPARSE_CONV_BACKEND"] = "spconv"      # Avoid Triton (broken on CC 12.0)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ── 1) Patch torch.cuda.get_device_capability ──
    _orig_cap = torch.cuda.get_device_capability
    def _patched_cap(device=None):
        m, n = _orig_cap(device)
        return (9, 0) if m >= 10 else (m, n)
    torch.cuda.get_device_capability = _patched_cap

    # ── 2) Patch cumm/spconv compute capability detection ──
    try:
        import cumm.tensorview as _tv
        _orig_cumm = _tv.get_compute_capability
        def _patched_cumm(index: int = -1):
            m, n = _orig_cumm(index)
            return (9, 0) if m >= 10 else (m, n)
        _tv.get_compute_capability = _patched_cumm

        try:
            import cumm.tensorview_bind as _tvb
            _orig_tvb = _tvb.get_compute_capability
            def _patched_tvb(index: int = -1):
                m, n = _orig_tvb(index)
                return (9, 0) if m >= 10 else (m, n)
            _tvb.get_compute_capability = _patched_tvb
        except (ImportError, AttributeError):
            pass

        if verbose:
            print("[blackwell_fix] Patched cumm CC detection -> (9, 0)")
    except ImportError:
        pass

    # ── 3) Patch flex_gemm Triton kernels with PyTorch fallbacks ──
    # Triton 3.3.x cannot compile for CC >= 10.0
    try:
        try:
            import flex_gemm.kernels.triton as _fgk
        except ImportError:
            _fgk = None

        def _fwd(feats, indices, weight):
            idx = indices.long().clamp(0, feats.shape[0] - 1)
            return (feats[idx] * weight.unsqueeze(-1)).sum(dim=1)

        def _bwd(grad_output, indices, weight, N):
            M, C = grad_output.shape
            idx = indices.long().clamp(0, N - 1)
            wg = grad_output.unsqueeze(1) * weight.unsqueeze(-1)
            gf = torch.zeros(N, C, device=grad_output.device, dtype=grad_output.dtype)
            gf.scatter_add_(0, idx.unsqueeze(-1).expand_as(wg).reshape(-1, C),
                            wg.reshape(-1, C))
            return gf

        _fgk.indice_weighed_sum_fwd = _fwd
        _fgk.indice_weighed_sum_bwd_input = _bwd
        if verbose:
            print("[blackwell_fix] Patched flex_gemm Triton kernels -> PyTorch")
    except (ImportError, AttributeError):
        pass

    # ── 4) Pillow WebP compatibility ──
    try:
        from PIL import _webp
        if not hasattr(_webp, 'HAVE_WEBPANIM'):
            _webp.HAVE_WEBPANIM = hasattr(_webp, 'WebPAnimDecoder')
        if not hasattr(_webp, 'HAVE_WEBPMUX'):
            _webp.HAVE_WEBPMUX = hasattr(_webp, 'WebPAnimDecoder')
        if not hasattr(_webp, 'HAVE_TRANSPARENCY'):
            _webp.HAVE_TRANSPARENCY = True
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Voxel-based mesh extraction (replaces broken CuMesh pipeline)
# ---------------------------------------------------------------------------

def voxel_to_mesh(
    mesh_output,
    target_height_mm: float = 100.0,
    sigma: float = 1.5,
    coarse_downsample: float = 4,
    taubin_iterations: int = 50,
    verbose: bool = True,
):
    """
    Convert a TRELLIS.2 mesh output to a watertight trimesh via voxel
    marching cubes. This replaces the broken to_glb() → CuMesh pipeline
    on Blackwell GPUs.

    Uses a two-phase approach:
      Phase 1: Coarse grid (downsampled) with aggressive morphological closing
               + flood fill to determine the solid interior.
      Phase 2: Full-resolution voxel grid combining surface voxels with the
               upscaled coarse interior. Gaussian smooth + marching cubes.

    Args:
        mesh_output: TRELLIS.2 mesh object (from pipeline.run()[0]).
                     Must have .coords (integer voxel positions) and
                     .voxel_size (float scaling factor).
        target_height_mm: Scale the output so the tallest dimension equals
                          this many millimeters. Set to 0 to skip scaling.
        sigma: Gaussian smoothing sigma for the volume before marching cubes.
               Higher = smoother surface but less detail. 1.0-2.0 recommended.
        coarse_downsample: Downsample factor for the coarse interior fill.
                           4 works well for most models.
        taubin_iterations: Number of Taubin mesh smoothing passes after
                           marching cubes. Reduces voxel staircase artifacts.
        verbose: Print progress messages.

    Returns:
        trimesh.Trimesh: Watertight mesh ready for export/printing.
    """
    import torch

    # Extract numpy arrays from the TRELLIS mesh object
    if isinstance(mesh_output.coords, torch.Tensor):
        coords_np = mesh_output.coords.cpu().numpy().copy()
    else:
        coords_np = np.array(mesh_output.coords).copy()
    voxel_size = float(mesh_output.voxel_size)

    return voxel_coords_to_mesh(
        coords_np=coords_np,
        voxel_size=voxel_size,
        target_height_mm=target_height_mm,
        sigma=sigma,
        coarse_downsample=coarse_downsample,
        taubin_iterations=taubin_iterations,
        verbose=verbose,
    )


def voxel_coords_to_mesh(
    coords_np: np.ndarray,
    voxel_size: float,
    target_height_mm: float = 100.0,
    sigma: float = 1.5,
    coarse_downsample: float = 4,
    taubin_iterations: int = 50,
    verbose: bool = True,
):
    """
    Convert raw voxel coordinates to a watertight trimesh.

    This is the lower-level version of voxel_to_mesh() that works directly
    with numpy arrays. Useful when you've already extracted and saved the
    coordinates (e.g., for iterating on mesh parameters without re-running
    inference).

    Args:
        coords_np: Integer voxel coordinates, shape (N, 3).
        voxel_size: Voxel size in model units.
        target_height_mm: Scale output height. Set to 0 to skip.
        sigma: Gaussian smoothing sigma.
        coarse_downsample: Coarse grid downsample factor.
        taubin_iterations: Mesh smoothing iterations.
        verbose: Print progress.

    Returns:
        trimesh.Trimesh: Watertight mesh.
    """
    import trimesh
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, zoom
    from skimage import measure
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    def log(msg):
        if verbose:
            print(f"  {msg}")

    t0 = time.time()
    log("Multi-resolution voxel reconstruction...")

    c_min = coords_np.min(axis=0)
    c_max = coords_np.max(axis=0)
    extent = (c_max - c_min).astype(int)
    log(f"Voxel extent: {extent[0]}x{extent[1]}x{extent[2]} "
        f"({coords_np.shape[0]:,} occupied, voxel_size={voxel_size:.6f})")

    struct26 = ndimage.generate_binary_structure(3, 3)  # 26-connected
    ds_c = coarse_downsample

    # ── Phase 1: Coarse fill for solid interior ──────────────────────────
    c_ds = ((coords_np - c_min) // ds_c).astype(int)
    grid_c = tuple(((c_max - c_min) // ds_c + 2).astype(int))
    vol_c = np.zeros(grid_c, dtype=np.uint8)
    vol_c[c_ds[:, 0], c_ds[:, 1], c_ds[:, 2]] = 1
    del c_ds
    log(f"Phase 1 — Coarse grid ({ds_c}x): {grid_c[0]}x{grid_c[1]}x{grid_c[2]}")

    # Aggressive morphological closing to seal all gaps
    vol_c = ndimage.binary_dilation(vol_c, struct26, iterations=5)
    pad_c = 3
    vol_c = np.pad(vol_c.astype(np.uint8), pad_c, mode='constant', constant_values=0)
    vol_c = ndimage.binary_fill_holes(vol_c)
    vol_c = ndimage.binary_erosion(vol_c, struct26, iterations=4)
    vol_c = vol_c[pad_c:-pad_c, pad_c:-pad_c, pad_c:-pad_c]
    log(f"Coarse interior: {int(vol_c.sum()):,} voxels")

    # ── Phase 2: Full-resolution grid + coarse interior ──────────────────
    pad_f = 2
    grid_f = tuple((c_max - c_min + 2 + 2 * pad_f).astype(int))
    log(f"Phase 2 — Fine grid (ds=1): {grid_f[0]}x{grid_f[1]}x{grid_f[2]} "
        f"({int(np.prod(grid_f))/1e6:.0f}M cells)")

    # Place surface voxels
    c_shifted = (coords_np - c_min + pad_f).astype(int)
    vol = np.zeros(grid_f, dtype=np.uint8)
    vol[c_shifted[:, 0], c_shifted[:, 1], c_shifted[:, 2]] = 1
    del c_shifted, coords_np
    gc.collect()
    surface = int(vol.sum())

    # Upscale coarse interior to fine resolution
    log("Upscaling coarse interior...")
    interior_up = zoom(vol_c.astype(np.float32), ds_c, order=3) > 0.3
    del vol_c
    gc.collect()

    # Align and merge (coarse interior may be slightly different shape)
    for d in range(3):
        if interior_up.shape[d] + pad_f > grid_f[d]:
            slc = [slice(None)] * 3
            slc[d] = slice(0, grid_f[d] - pad_f)
            interior_up = interior_up[tuple(slc)]
    slices = tuple(slice(pad_f, pad_f + interior_up.shape[d]) for d in range(3))
    vol[slices] |= interior_up.astype(np.uint8)
    del interior_up
    gc.collect()
    total = int(vol.sum())
    log(f"Surface: {surface:,} voxels, with interior: {total:,}")

    # ── Gaussian smooth + marching cubes ─────────────────────────────────
    log(f"Gaussian smoothing (sigma={sigma})...")
    vol = vol.astype(np.float32)
    gc.collect()
    gaussian_filter(vol, sigma=sigma, output=vol)  # in-place for memory

    log("Running marching cubes...")
    verts, faces, normals, _ = measure.marching_cubes(vol, level=0.5)
    del vol
    gc.collect()
    log(f"Marching cubes: {len(verts):,} vertices, {len(faces):,} faces")

    # Scale back to model coordinates
    verts = (verts - pad_f) + c_min
    verts = verts * voxel_size

    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # ── Keep largest connected component ─────────────────────────────────
    adj = tm.face_adjacency
    nf = len(tm.faces)
    if len(adj) > 0:
        row = np.concatenate([adj[:, 0], adj[:, 1]])
        col = np.concatenate([adj[:, 1], adj[:, 0]])
        graph = coo_matrix(
            (np.ones(len(row), dtype=np.int32), (row, col)), shape=(nf, nf)
        )
        nc, labels = connected_components(graph, directed=False)
        if nc > 1:
            from collections import Counter
            largest = Counter(labels).most_common(1)[0][0]
            keep = np.where(labels == largest)[0]
            tm = tm.submesh([keep], append=True)
            log(f"Kept largest of {nc} components")

    # ── Mesh smoothing ───────────────────────────────────────────────────
    if taubin_iterations > 0:
        log(f"Taubin mesh smoothing ({taubin_iterations} iterations)...")
        trimesh.smoothing.filter_taubin(tm, iterations=taubin_iterations)

    # ── Scale to physical size ───────────────────────────────────────────
    if target_height_mm > 0:
        scale_factor = target_height_mm / tm.extents.max()
        tm.apply_scale(scale_factor)
        log(f"Scaled to {target_height_mm:.0f}mm "
            f"({tm.extents[0]:.1f} x {tm.extents[1]:.1f} x {tm.extents[2]:.1f})")

    tm.fix_normals()
    log(f"Watertight: {tm.is_watertight}")
    log(f"Final: {len(tm.vertices):,} vertices, {len(tm.faces):,} faces")
    log(f"Completed in {time.time() - t0:.1f}s")

    return tm


# ---------------------------------------------------------------------------
# Convenience: run mesh reconstruction in a subprocess (avoids OOM)
# ---------------------------------------------------------------------------

def voxel_to_mesh_subprocess(
    mesh_output,
    output_path: str,
    target_height_mm: float = 100.0,
    sigma: float = 1.5,
    timeout: int = 600,
    verbose: bool = True,
) -> bool:
    """
    Run voxel_to_mesh in a separate process to avoid OOM.

    TRELLIS.2 inference retains significant CPU memory even after the model
    is deleted. Running the mesh reconstruction in a subprocess starts with
    clean memory, which is necessary for the full-resolution (ds=1) grid
    on systems with <= 32GB RAM.

    Args:
        mesh_output: TRELLIS.2 mesh object.
        output_path: Where to save the mesh (STL, OBJ, PLY, etc.).
        target_height_mm: Scale output height in mm.
        sigma: Gaussian smoothing sigma.
        timeout: Max seconds for subprocess.
        verbose: Print progress.

    Returns:
        True if the output file was created successfully.
    """
    import torch
    import subprocess
    import tempfile

    # Save coords to temp file
    if isinstance(mesh_output.coords, torch.Tensor):
        coords_np = mesh_output.coords.cpu().numpy().copy()
    else:
        coords_np = np.array(mesh_output.coords).copy()
    voxel_size = float(mesh_output.voxel_size)

    coords_file = tempfile.mktemp(suffix='_coords.npy')
    np.save(coords_file, coords_np)
    del coords_np

    try:
        result = subprocess.run([
            sys.executable, '-c',
            f"""
import numpy as np
from blackwell_fix import voxel_coords_to_mesh
coords = np.load({coords_file!r})
tm = voxel_coords_to_mesh(coords, {voxel_size}, target_height_mm={target_height_mm}, sigma={sigma})
tm.export({output_path!r})
print(f"Saved: {{output_path}}")
""",
        ], timeout=timeout)
        return os.path.exists(output_path)
    finally:
        try:
            os.remove(coords_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert saved TRELLIS.2 voxel coords to watertight mesh"
    )
    parser.add_argument("coords_file", help="Path to .npy file with voxel coordinates")
    parser.add_argument("voxel_size", type=float, help="Voxel size from mesh.voxel_size")
    parser.add_argument("-o", "--output", default="output.stl", help="Output mesh path")
    parser.add_argument("--height", type=float, default=100.0, help="Target height in mm")
    parser.add_argument("--sigma", type=float, default=1.5, help="Gaussian smoothing sigma")
    parser.add_argument("--taubin", type=int, default=50, help="Taubin smoothing iterations")
    args = parser.parse_args()

    coords = np.load(args.coords_file)
    tm = voxel_coords_to_mesh(
        coords, args.voxel_size,
        target_height_mm=args.height,
        sigma=args.sigma,
        taubin_iterations=args.taubin,
    )
    tm.export(args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved {args.output} ({size_mb:.1f} MB)")
