from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class DinoLockMixin:
    """
    Shared DINO-lock functionality for any GuidanceInterval sampler.

    When ``dino_lock > 0`` each step computes both the CFG-guided velocity
    and the pure positive-conditioned (DINO-only) velocity, then blends
    toward the DINO direction.  The schedule builds the initial shape from
    DINOv3 features first, then hands off to CFG for detail:

        +-----------+---------------------------+
        | Steps     | Lock strength             |
        +-----------+---------------------------+
        | 0 – 40 %  | 0.92 (full DINO: shape)   |
        | 40 – 70 % | ramp 0.92 → dino_lock     |
        | 70 – 100 %| dino_lock (CFG guardrail)  |
        +-----------+---------------------------+

    The mixin intercepts ``sample()`` to add the ``dino_lock`` and
    ``dino_substeps`` keyword arguments.  When ``dino_lock <= 0`` it
    falls straight through to the underlying sampler's ``sample()``.
    """

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _get_pos_only_v(self, model, x_t, t, cond, **kwargs):
        """Run model with guidance_strength=1 (positive cond only)."""
        pos_kw = dict(kwargs)
        pos_kw["guidance_strength"] = 1.0
        return self._inference_model(model, x_t, t, cond, **pos_kw)

    def _dino_project(self, guided_v, pos_v, lock_strength):
        """
        Linear velocity blend toward the DINO-only signal.

        lock_strength=0 → guided_v unchanged
        lock_strength=1 → pure pos_v (full DINO trajectory)
        """
        return (1.0 - lock_strength) * guided_v + lock_strength * pos_v

    @staticmethod
    def _alignment_stats(guided_v, pos_v):
        """Alignment statistics between guided and positive-only velocity."""
        g_raw = guided_v.feats if hasattr(guided_v, 'feats') else guided_v
        p_raw = pos_v.feats if hasattr(pos_v, 'feats') else pos_v

        g_flat = g_raw.reshape(-1).float().unsqueeze(0)
        p_flat = p_raw.reshape(-1).float().unsqueeze(0)

        g_norm = g_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        p_norm = p_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

        cos = (g_flat * p_flat).sum(dim=1) / (g_norm.squeeze() * p_norm.squeeze())
        cos_mean = cos.mean().item()
        angle = np.degrees(np.arccos(np.clip(cos_mean, -1.0, 1.0)))
        mag_ratio = (p_norm.squeeze() / g_norm.squeeze()).mean().item()
        drift = (g_flat - p_flat).norm(dim=1).mean().item() / g_norm.squeeze().mean().item()

        proj_c = (g_flat * p_flat).sum(dim=1, keepdim=True) / (p_flat * p_flat).sum(dim=1, keepdim=True).clamp(min=1e-8)
        perp = g_flat - proj_c * p_flat
        perp_ratio = perp.norm(dim=1).mean().item() / g_norm.squeeze().mean().item()

        return {"cos_sim": cos_mean, "mag_ratio": mag_ratio,
                "angle_deg": angle, "drift": drift, "perp_ratio": perp_ratio}

    def _dino_lock_step(self, model, x_t, t, t_prev, cond,
                         lock_strength, step_idx, total_steps,
                         substeps=1, v_ema=None, ema_alpha=0.85, verbose=True, **kwargs):
        """
        One step with DINO lock + velocity EMA smoothing.

        v_ema smoothing reduces velocity discontinuities at phase
        transitions: v_final = α·v_current + (1-α)·v_ema_prev.
        """
        guided_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pos_v = self._get_pos_only_v(model, x_t, t, cond, **kwargs)

        stats = self._alignment_stats(guided_v, pos_v)
        
        if verbose:
            phase = "FOUND" if lock_strength >= 0.9 else ("RAMP" if step_idx >= int(total_steps * 0.4) and step_idx < int(total_steps * 0.7) else "GUARD")
            sub_tag = f" x{substeps}" if substeps > 1 else ""
            ema_tag = " +ema" if v_ema is not None else ""
            print(f"  [DinoLock {step_idx+1:>3}/{total_steps}] "
                  f"cos={stats['cos_sim']:+.4f}  "
                  f"angle={stats['angle_deg']:5.1f}°  "
                  f"perp={stats['perp_ratio']:.3f}  "
                  f"drift={stats['drift']:.4f}  "
                  f"lock={lock_strength:.3f} ({phase}{sub_tag}{ema_tag})")

        if lock_strength <= 0.0:
            pred_v = guided_v
        else:
            pred_v = self._dino_project(guided_v, pos_v, lock_strength)

        if v_ema is not None:
            pred_v = ema_alpha * pred_v + (1.0 - ema_alpha) * v_ema
        new_v_ema = pred_v

        if substeps > 1 and lock_strength >= 0.9:
            dt_total = t - t_prev
            dt_sub = dt_total / substeps
            current = x_t
            t_cur = t
            for _s in range(substeps):
                v_sub = self._inference_model(model, current, t_cur, cond, **kwargs)
                current = current - dt_sub * v_sub
                t_cur = t_cur - dt_sub
            pred_x_prev = current
        else:
            pred_x_prev = x_t - (t - t_prev) * pred_v

        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0,
                       "stats": stats, "lock_strength": lock_strength,
                       "v_ema": new_v_ema})

    @staticmethod
    def _compute_lock_strength(step_idx, total_steps, base_strength, foundation_cap = 0.92):
        """
        DINO Foundation schedule:
        - Steps  0 – 40 %:  0.92  (near-full DINO, 8 % CFG on-distribution).
        - Steps 40 – 70 %:  cosine ramp 0.92 → base_strength.
        - Steps 70 – 100 %: base_strength (residual guardrail).
        """
        #FOUNDATION_CAP = 0.92
        foundation_end = int(total_steps * 0.4)
        ramp_end = int(total_steps * 0.7)
        if step_idx < foundation_end:
            return foundation_cap
        if step_idx >= ramp_end:
            return base_strength
        progress = (step_idx - foundation_end) / max(1, ramp_end - foundation_end)
        blend = 0.5 * (1.0 - np.cos(np.pi * progress))
        return float(foundation_cap + (base_strength - foundation_cap) * blend)

    # ------------------------------------------------------------------ #
    #  sample() override                                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond=None,
        neg_cond=None,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,        
        **kwargs
    ):
        # Strip keys that must not reach the model
        kwargs.pop("rk4_cond_lock_strength", None)
        kwargs.pop("rk4_cond_lock_end_strength", None)
        kwargs.pop("debug_dino_alignment", None)
        kwargs.pop("debug_dino_interval", None)

        # Multiview path: cond/neg_cond not provided (per-view conds are in kwargs)
        if cond is None:
            return super().sample(model, noise,
                                  steps=steps, rescale_t=rescale_t, verbose=verbose,
                                  tqdm_desc=tqdm_desc,
                                  guidance_strength=guidance_strength,
                                  guidance_interval=guidance_interval,
                                  **kwargs)

        if dino_lock <= 0.0:
            return super().sample(model, noise, cond, 
                                  steps = steps, 
                                  rescale_t = rescale_t, 
                                  verbose = verbose,
                                  tqdm_desc=tqdm_desc,
                                  neg_cond=neg_cond,
                                  guidance_strength=guidance_strength,
                                  guidance_interval=guidance_interval,
                                  **kwargs)

        # ----- DINO-locked sampling loop -----
        if verbose:
            print(f"\n{'='*72}")
            print(f"  DINO Foundation  |  guardrail={dino_lock:.2f}  |  steps={steps}")
            print(f"  Schedule: 0-40% DINO@0.92 (shape), 40-70% ramp→{dino_lock:.2f}, 70-100% guardrail")
            print(f"  Mode: foundation-first + velocity EMA smoothing")
            if dino_substeps > 1:
                print(f"  Substeps: {dino_substeps}x during foundation phase")
            print(f"{'='*72}")

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

        merged_kwargs = dict(kwargs)
        merged_kwargs["neg_cond"] = neg_cond
        merged_kwargs["guidance_strength"] = guidance_strength
        merged_kwargs["guidance_interval"] = guidance_interval

        all_stats = []
        v_ema = None
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for i, (t, t_prev) in enumerate(tqdm(t_pairs, desc=tqdm_desc)):
            s = self._compute_lock_strength(i, steps, dino_lock, dino_foundation_cap)
            n_sub = dino_substeps if (s >= 0.9 and dino_substeps > 1) else 1
            out = self._dino_lock_step(model, sample, t, t_prev, cond,
                                        lock_strength=s,
                                        step_idx=i, total_steps=steps,
                                        substeps=n_sub,
                                        v_ema=v_ema,
                                        verbose=verbose,
                                        **merged_kwargs)
            sample = out.pred_x_prev
            v_ema = out.v_ema
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
            all_stats.append(out.stats)

        avg_cos = np.mean([s["cos_sim"] for s in all_stats])
        avg_drift = np.mean([s["drift"] for s in all_stats])
        avg_perp = np.mean([s["perp_ratio"] for s in all_stats])
        final_cos = all_stats[-1]["cos_sim"]
        final_angle = all_stats[-1]["angle_deg"]
        final_perp = all_stats[-1]["perp_ratio"]
        
        if verbose:
            print(f"\n{'─'*72}")
            print(f"  DINO Foundation Summary")
            print(f"  avg cos_sim={avg_cos:+.4f}  avg drift={avg_drift:.4f}  avg perp={avg_perp:.4f}")
            print(f"  final cos_sim={final_cos:+.4f}  final angle={final_angle:.1f}°  final perp={final_perp:.4f}")
            print(f"{'─'*72}\n")

        ret.samples = sample
        return ret


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
    
    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """Euler sampling with CFG, guidance interval, and optional DINO lock."""
    pass


class FlowEulerMultiViewSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with multi-view blending.
    """
    def __init__(self, sigma_min: float, resolution: int):
        super().__init__(sigma_min)
        self.resolution = resolution
    
    def _compute_view_weights_sparse(self, coords, views, front_axis='z', blend_temperature=2.0) -> torch.Tensor:
        """
        Compute blending weights for sparse voxels.
        """
        # Normalize coords to [-1, 1] range (roughly)
        z = (coords[:, 1].float() / self.resolution) * 2 - 1.0
        x = (coords[:, 3].float() / self.resolution) * 2 - 1.0
        
        if front_axis == 'z':
            # Front (+Z), Back (-Z), Right (+X), Left (-X)
            view_vectors = {
                'front': torch.stack([torch.zeros_like(z), z], dim=1), # (0, z)
                'back':  torch.stack([torch.zeros_like(z), -z], dim=1),
                'right': torch.stack([x, torch.zeros_like(x)], dim=1),
                'left':  torch.stack([-x, torch.zeros_like(x)], dim=1),
            }
        else: # front_axis == 'x' (swapped)
            # Front (+X), Back (-X), Right (+Z), Left (-Z)
             view_vectors = {
                'front': torch.stack([x, torch.zeros_like(x)], dim=1),
                'back':  torch.stack([-x, torch.zeros_like(x)], dim=1),
                'right': torch.stack([torch.zeros_like(z), z], dim=1),
                'left':  torch.stack([torch.zeros_like(z), -z], dim=1),
            }

        scores = []
        for view in views:
            if view in view_vectors:
                v_vec = view_vectors[view]
                score = v_vec.sum(dim=1)
                scores.append(score)
            else:
                scores.append(torch.full_like(z, -10.0))
        
        scores = torch.stack(scores, dim=1) # (N, num_views)
        weights = torch.softmax(scores * blend_temperature, dim=1)
        return weights

    def _compute_view_weights_dense(self, shape, device, views, front_axis='z', blend_temperature=2.0) -> torch.Tensor:
        """
        Compute blending weights for dense grid (B, C, D, H, W).
        Returns weights of shape (1, 1, D, H, W, NumViews) for easy broadcasting (actually we want (1, 1, D, H, W) per view)
        """
        # shape is (B, C, D, H, W)
        D, H, W = shape[2], shape[3], shape[4]
        
        # Create meshgrid in [-1, 1]
        # We assume D is Z axis, W is X axis (usually D, H, W = Z, Y, X in 3D tensors?)
        # Let's verify standard: (Batch, Channel, Depth, Height, Width) -> (B, C, Z, Y, X)
        
        dz = torch.linspace(-1, 1, D, device=device)
        dy = torch.linspace(-1, 1, H, device=device)
        dx = torch.linspace(-1, 1, W, device=device)
        
        # meshgrid 'ij' indexing: (D, H, W) order
        grid_z, grid_y, grid_x = torch.meshgrid(dz, dy, dx, indexing='ij') 
        
        # Flatten for vector calc? Or keep structural. Keep structural.
        
        if front_axis == 'z':
             # Front (+Z), Back (-Z), Right (+X), Left (-X)
             # Vectors are scalar fields here
             view_scores = {
                 'front': grid_z,
                 'back': -grid_z,
                 'right': grid_x,
                 'left': -grid_x,
             }
        else:
             view_scores = {
                 'front': grid_x,
                 'back': -grid_x,
                 'right': grid_z,
                 'left': -grid_z,
             }
             
        scores = []
        for view in views:
            if view in view_scores:
                scores.append(view_scores[view])
            else:
                scores.append(torch.full_like(grid_z, -10.0))
                
        # Stack: (NumViews, D, H, W)
        scores = torch.stack(scores, dim=0) 
        
        # Softmax over views dimension (0)
        weights = torch.softmax(scores * blend_temperature, dim=0)
        
        # Reshape for broadcasting: (NumViews, 1, 1, D, H, W) -> No wait, loop is over views.
        # We want to return something we can index like weights[i] -> (1, 1, D, H, W)
        
        # Current shape: (NumViews, D, H, W)
        return weights

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        conds: Dict[str, Any], # Changed: expects dict of {view: cond}
        views: List[str],      # Changed: list of view keys corresponding to conds
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        **kwargs
    ):
        """
        Sample with multi-view blending.
        """
        is_sparse = hasattr(x_t, 'coords')
        
        if is_sparse:
            # 1. Compute per-voxel weights based on current sparse coords
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
            # weights: (N, NumViews)
        else:
            # Dense tensor (B, C, D, H, W)
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            # weights: (NumViews, D, H, W)
        
        # 2. Run model for each view and blend predictions
        pred_v_accum = 0
        
        for i, view in enumerate(views):
            cond = conds[view]
            # Use _inference_model to support mixins (CFG, etc)
            # If cond is a dict containing 'cond' and 'neg_cond' (from pipeline.get_cond), unpack it
            if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                pred_v_view = self._inference_model(model, x_t, t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
            else:
                pred_v_view = self._inference_model(model, x_t, t, cond=cond, **kwargs)
            
            # Weighted accumulation
            if is_sparse:
                # weights[:, i] is (N,), pred_v_view might be SparseTensor or Tensor (N, C)
                w = weights[:, i].unsqueeze(1)
                
                v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                pred_v_accum += v_feats * w
            else:
                # Dense
                # weights[i] is (D, H, W). pred_v_view is (B, C, D, H, W)
                w = weights[i].unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
                pred_v_accum += pred_v_view * w
                
        if is_sparse:
            # Re-wrap accumulated features into a SparseTensor matching x_t
            # pred_v_accum is (N, C) tensor now
            pred_v = x_t.replace(feats=pred_v_accum)
        else:
            pred_v = pred_v_accum
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)

        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        conds: Dict[str, Any], # {view: cond}
        views: List[str],      # ['front', 'back', ...]
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling MultiView",
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        **kwargs
    ):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc):
            out = self.sample_once(
                model, sample, t, t_prev, 
                conds=conds, 
                views=views,
                front_axis=front_axis, 
                blend_temperature=blend_temperature, 
                **kwargs
            )
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerMultiViewGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerMultiViewSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with multi-view blending, CFG, and guidance interval.
    """
    pass
    
# RK4 and RK5 Samplers

class FlowRK4Sampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using the 4th-order Runge-Kutta method.
    """
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        dt = t_prev - t
        
        # Helper to extract just the velocity prediction
        def get_v(current_x, current_t):
            _, _, pred_v = self._get_model_prediction(model, current_x, current_t, cond, **kwargs)
            return pred_v

        # RK4 intermediate slopes
        k1 = get_v(x_t, t)
        k2 = get_v(x_t + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = get_v(x_t + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = get_v(x_t + dt * k3, t + dt)
        
        # RK4 integration
        pred_x_prev = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # We need to return pred_x_0 as well to satisfy the pipeline's logging/tracking
        # We compute x_start_eps based on the k1 velocity (equivalent to the Euler estimation of x_0)
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowRK5Sampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Butcher's 5th-order Runge-Kutta method.
    """
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        dt = t_prev - t
        
        # Helper to extract just the velocity prediction
        def get_v(current_x, current_t):
            _, _, pred_v = self._get_model_prediction(model, current_x, current_t, cond, **kwargs)
            return pred_v

        # Intermediate time step fractions for Butcher's RK5
        c2, c3, c4, c5, c6 = 1/4, 1/4, 1/2, 3/4, 1.0
        
        k1 = get_v(x_t, t)
        k2 = get_v(x_t + dt * (1/4 * k1), t + dt * c2)
        k3 = get_v(x_t + dt * (1/8 * k1 + 1/8 * k2), t + dt * c3)
        k4 = get_v(x_t + dt * (-1/2 * k2 + 1.0 * k3), t + dt * c4)
        k5 = get_v(x_t + dt * (3/16 * k1 + 9/16 * k4), t + dt * c5)
        k6 = get_v(x_t + dt * (-3/7 * k1 + 2/7 * k2 + 12/7 * k3 - 12/7 * k4 + 8/7 * k5), t + dt * c6)
        
        # Final RK5 Integration
        pred_x_prev = x_t + dt * (7/90 * k1 + 32/90 * k3 + 12/90 * k4 + 32/90 * k5 + 7/90 * k6)
        
        # Estimate x_0 based on k1 for tracking
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


# --- Classifier Free Guidance (CFG) Wrappers ---
class FlowRK4CfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowRK4Sampler):
    """RK4 sampling with classifier-free guidance."""
    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0, guidance_strength: float = 3.0, verbose: bool = True, **kwargs):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)

class FlowRK5CfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowRK5Sampler):
    """RK5 sampling with classifier-free guidance."""
    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0, guidance_strength: float = 3.0, verbose: bool = True, **kwargs):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)
        
class FlowRK4GuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK4Sampler):
    """RK4 with CFG, Guidance Intervals, and optional DINO lock."""
    pass

class FlowRK5GuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK5Sampler):
    """RK5 with CFG, Guidance Intervals, and optional DINO lock."""
    pass        
    
# RK4 and RK5 for MultiView

class FlowRK4MultiViewSampler(FlowEulerMultiViewSampler):
    """Multi-view flow matching using 4th-order Runge-Kutta."""
    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, 
        conds: Dict[str, Any], views: List[str], 
        front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs
    ):
        dt = t_prev - t
        is_sparse = hasattr(x_t, 'coords')
        
        # Calculate spatial blending weights ONCE for the current step
        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            
        # Helper function to compute the blended velocity for a given intermediate x and t
        def get_blended_v(current_x, current_t):
            pred_v_accum = 0
            for i, view in enumerate(views):
                cond = conds[view]
                if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
                else:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond, **kwargs)
                
                if is_sparse:
                    w = weights[:, i].unsqueeze(1)
                    v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                    pred_v_accum += v_feats * w
                else:
                    w = weights[i].unsqueeze(0).unsqueeze(0)
                    pred_v_accum += pred_v_view * w
                    
            if is_sparse:
                return current_x.replace(feats=pred_v_accum)
            else:
                return pred_v_accum

        # RK4 Evaluations
        k1 = get_blended_v(x_t, t)
        k2 = get_blended_v(x_t + k1 * (0.5 * dt), t + 0.5 * dt)
        k3 = get_blended_v(x_t + k2 * (0.5 * dt), t + 0.5 * dt)
        k4 = get_blended_v(x_t + k3 * dt, t + dt)
        
        pred_x_prev = x_t + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6.0)
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowRK5MultiViewSampler(FlowEulerMultiViewSampler):
    """Multi-view flow matching using Butcher's 5th-order Runge-Kutta."""
    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, 
        conds: Dict[str, Any], views: List[str], 
        front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs
    ):
        dt = t_prev - t
        is_sparse = hasattr(x_t, 'coords')
        
        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            
        def get_blended_v(current_x, current_t):
            pred_v_accum = 0
            for i, view in enumerate(views):
                cond = conds[view]
                if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
                else:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond, **kwargs)
                
                if is_sparse:
                    w = weights[:, i].unsqueeze(1)
                    v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                    pred_v_accum += v_feats * w
                else:
                    w = weights[i].unsqueeze(0).unsqueeze(0)
                    pred_v_accum += pred_v_view * w
                    
            if is_sparse:
                return current_x.replace(feats=pred_v_accum)
            else:
                return pred_v_accum

        # Butcher Tableau Intermediate steps
        c2, c3, c4, c5, c6 = 1/4, 1/4, 1/2, 3/4, 1.0
        
        k1 = get_blended_v(x_t, t)
        k2 = get_blended_v(x_t + k1 * (dt * 1/4), t + dt * c2)
        k3 = get_blended_v(x_t + (k1 * 1/8 + k2 * 1/8) * dt, t + dt * c3)
        k4 = get_blended_v(x_t + (k2 * -1/2 + k3 * 1.0) * dt, t + dt * c4)
        k5 = get_blended_v(x_t + (k1 * 3/16 + k4 * 9/16) * dt, t + dt * c5)
        k6 = get_blended_v(x_t + (k1 * -3/7 + k2 * 2/7 + k3 * 12/7 + k4 * -12/7 + k5 * 8/7) * dt, t + dt * c6)
        
        pred_x_prev = x_t + (k1 * 7/90 + k3 * 32/90 + k4 * 12/90 + k5 * 32/90 + k6 * 7/90) * dt
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowRK4MultiViewGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK4MultiViewSampler):
    pass

class FlowRK5MultiViewGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK5MultiViewSampler):
    pass

# Heun (RK2)

class FlowHeunSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Heun's Method (2nd-order Runge-Kutta).
    Requires 2 NFEs per step.
    """
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        dt = t_prev - t
        
        # Helper to extract just the velocity prediction
        def get_v(current_x, current_t):
            _, _, pred_v = self._get_model_prediction(model, current_x, current_t, cond, **kwargs)
            return pred_v

        # Step 1: Predictor (Euler step)
        k1 = get_v(x_t, t)
        x_temp = x_t + k1 * dt
        
        # Step 2: Corrector
        k2 = get_v(x_temp, t + dt)
        
        # Average the two velocities for the final step
        pred_x_prev = x_t + 0.5 * dt * (k1 + k2)
        
        # Estimate x_0 based on k1 for tracking/logging
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

# --- CFG Wrapper for Heun ---
class FlowHeunGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowHeunSampler):
    """Heun sampling with CFG, Guidance Intervals, and optional DINO lock."""
    pass
    
class FlowHeunMultiViewSampler(FlowEulerMultiViewSampler):
    """Multi-view flow matching using Heun's method (2nd-order Runge-Kutta)."""
    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, 
        conds: Dict[str, Any], views: List[str], 
        front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs
    ):
        dt = t_prev - t
        is_sparse = hasattr(x_t, 'coords')
        
        # Calculate spatial blending weights ONCE for the current step
        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            
        # Helper function to compute the blended velocity for a given intermediate x and t
        def get_blended_v(current_x, current_t):
            pred_v_accum = 0
            for i, view in enumerate(views):
                cond = conds[view]
                if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
                else:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond, **kwargs)
                
                if is_sparse:
                    w = weights[:, i].unsqueeze(1)
                    v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                    pred_v_accum += v_feats * w
                else:
                    w = weights[i].unsqueeze(0).unsqueeze(0)
                    pred_v_accum += pred_v_view * w
                    
            if is_sparse:
                return current_x.replace(feats=pred_v_accum)
            else:
                return pred_v_accum

        # Heun's Method (RK2) Evaluations
        # Step 1: Predictor (Euler step)
        k1 = get_blended_v(x_t, t)
        x_temp = x_t + k1 * dt
        
        # Step 2: Corrector
        k2 = get_blended_v(x_temp, t + dt)
        
        # Combine
        pred_x_prev = x_t + 0.5 * dt * (k1 + k2)
        
        # Estimate x_0 based on k1 for tracking
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

# --- CFG Wrapper for Heun Multi-View ---
class FlowHeunMultiViewGuidanceIntervalSampler(DinoLockMixin, GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowHeunMultiViewSampler):
    pass    